"""Live, interactive world-model demo: drive MIRA-Mini with the keyboard and watch it generate in
real time, toggling between our optimized inference (2-step PSD + torch.compile + CUDA graphs) and the
released baseline (10-step base, eager) — with a live, server-measured speed/cost HUD.

It is legit end-to-end: a real H100 holds both real checkpoints warm, bootstraps from a real
rocket-science clip, and streams actual generated frames as you press keys. The HUD numbers are
measured on the GPU per frame (torch.cuda.Event), not scripted. Transport is HTTP polling (one POST
per frame) — robust on Modal's runtime; the per-frame GPU time is what the HUD reports.

    modal serve scripts/live_demo.py     # dev: prints a *.modal.run URL, open it, drive with WASD
    modal deploy scripts/live_demo.py    # persistent URL (run `modal app stop mira-live-demo` when done)

Controls: W/A/S/D drive, Q/E air-roll, Space ball-cam, Shift boost/powerslide, Ctrl. Buttons toggle
Ours vs Released and reset the world. See docs/optimization_plan.md (E2) for the ~7x result.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/mira"

# Checkpoints staged on the volume by modal_bench.py::stage_assets.
PSD_CKPT = "/data/checkpoints/mira-mini-psd/checkpoint-10000/checkpoint.pth"
BASE_CKPT = "/data/checkpoints/mira-mini/checkpoint-52000/checkpoint.pth"
DATA_INDEX = "/data/datasets/rocket-science/test"

# Modal H100 on-demand list price ($/hr) for the frames-per-dollar HUD. Update if your rate differs;
# it only scales the cost readout, not the (measured) speed. https://modal.com/pricing
H100_USD_PER_HR = 3.95


def _prefetch_dino() -> None:
    import torch

    # trust_repo + skip_validation avoid torch.hub's GitHub *API* fork-validation call, which is
    # unauthenticated (60 req/hr) and 403s on rebuilds ("rate limit exceeded"). The zipball still
    # downloads (different endpoint); once cached, runtime is a cache hit with no network/validation.
    torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", source="github",
                   verbose=False, pretrained=False, trust_repo=True, skip_validation=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "ffmpeg")
    .pip_install("torch==2.8.*", "torchvision==0.23.*",
                 extra_index_url="https://download.pytorch.org/whl/cu128")
    .run_commands("pip install --no-deps torchcodec==0.7.0 --index-url https://download.pytorch.org/whl/cpu")
    .add_local_dir(str(REPO_ROOT), REMOTE_REPO, copy=True,
                   ignore=["**/.git", "**/.pixi", "**/__pycache__", "**/*.lock", "**/.pytest_cache",
                           "**/.benchmarks"])
    .run_commands(f"pip install -e '{REMOTE_REPO}[train]'")
    .pip_install("fastapi", "uvicorn[standard]")
    .run_function(_prefetch_dino)
)

data_vol = modal.Volume.from_name("mira-bench-data", create_if_missing=True)
app = modal.App("mira-live-demo", image=image)


# --------------------------------------------------------------------------- inference session


class LiveSession:
    """One interactive rollout: bootstraps from a real clip, then steps frame-by-frame on live actions.

    Holds the sliding latent window + streaming KV-cache + (optional) CUDA-graph runner + action
    history for the currently selected mode. `step(keys)` denoises one latent, decodes it, and returns
    the newest video frame(s) plus the measured GPU time.
    """

    def __init__(self, models: dict, seed_batches: list, device, mode: str = "optimized"):
        import torch  # noqa: PLC0415

        self.torch = torch
        self.models = models          # {"optimized": (model, cfg), "baseline": (model, cfg)}
        self.seed_batches = seed_batches
        self.device = device
        self.seed_idx = 0
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """Switch optimized/baseline and re-bootstrap the world from the current seed clip."""
        if mode not in self.models:
            mode = "optimized"
        self.mode = mode
        self.model, self.cfg = self.models[mode]
        self.reset()

    def reset(self) -> None:
        torch = self.torch
        model = self.model
        seed = self.seed_batches[self.seed_idx % len(self.seed_batches)]
        self.window = model.n_context_latents + 1
        self.td = model.temporal_downsampling
        z_full = model.init_streaming_inference(seed.clone())
        # Start from the last `window` latents so the streaming state matches the trained window size.
        self.z = z_full[:, -self.window:].contiguous()
        self.kv = None
        # Action history: the seed's actions (aligned to the context frames). We append live actions.
        self.actions = seed.actions.clone().to(self.device)
        self.n_keys = self.actions.key_presses.shape[-1]
        self.graph_runner = self._make_graph_runner()
        self.frame_idx = 0

    def next_seed(self) -> None:
        self.seed_idx += 1
        self.reset()

    def _make_graph_runner(self):
        """Build a FrameGraphRunner for the optimized path (cuda_graphs on); None otherwise."""
        if not self.cfg.cuda_graphs:
            return None
        from mira.inference.cuda_graphs import FrameGraphRunner  # noqa: PLC0415

        return FrameGraphRunner(self.model, self.cfg.n_diffusion_steps, self.cfg.noise_level,
                                self.cfg.schedule_type)

    def _append_action(self, keys: list[int]) -> None:
        """Append the currently-held multi-hot key vector for this step's `td` video frames."""
        torch = self.torch
        vec = torch.zeros(self.n_keys, dtype=torch.int32, device=self.device)
        for k in keys:
            if 0 <= k < self.n_keys:
                vec[k] = 1
        # td actions per latent frame (the player holds one control across the chunk).
        new = self.actions.slice_time(0, 0)  # empty clone with same config/batch/device
        new.key_presses = vec.view(1, 1, self.n_keys).repeat(1, self.td, 1)
        # Rocket-science is keyboard-only; keep mouse zeros but time-aligned so cat_time/n_steps agree.
        new.mouse_movements = torch.zeros((1, self.td, 2), dtype=torch.float32, device=self.device)
        self.actions = self.actions.cat_time(new)

    def step(self, keys: list[int]):
        """Denoise one latent from the held keys, decode it, and return (uint8 HWC frame, denoise_ms).

        `denoise_ms` times ONLY the diffusion (streaming_inference_step) -- the thing we optimized
        (2-step vs 10-step) and the honest Ours-vs-Released number; decode is a shared constant cost.
        """
        torch = self.torch
        self._append_action(keys)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            start.record()
            self.z, self.kv = self.model.streaming_inference_step(
                self.z, self.actions, streaming_kv_cache=self.kv, config=self.cfg,
                graph_runner=self.graph_runner, ring_cache=self.cfg.cuda_graphs,
            )
            end.record()
            torch.cuda.synchronize()
            denoise_ms = start.elapsed_time(end)
            # Decode a short trailing window: the temporally-downsampled codec needs neighbouring
            # latents for a valid frame -- decoding a single latent in isolation renders black.
            dw = min(4, self.z.shape[1])
            video = self.model.decode_to_video(self.z[:, -dw:])  # (1, dw*td, C, H, W) in [-1, 1]

        vid = ((video[0, -1].clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8)  # (C, H, W)
        frame = vid.permute(1, 2, 0).contiguous().cpu().numpy()                # (H, W, C)
        if self.frame_idx < 3:  # diagnostic: confirm the decode carries signal (not all-zero/black)
            print(f"[{self.mode} f{self.frame_idx}] denoise {denoise_ms:.1f}ms  decoded "
                  f"min/max/mean={video.min():.2f}/{video.max():.2f}/{video.mean():.2f}  "
                  f"frame={tuple(frame.shape)}", flush=True)
        self.frame_idx += 1
        # Trim the action history so it can't grow unbounded over a long session.
        keep = (self.window + 2) * self.td
        if self.actions.key_presses.shape[1] > keep:
            self.actions = self.actions.slice_time(-keep, None)
        return frame, denoise_ms


# --------------------------------------------------------------------------- model loading


def _load_models(device):
    """Load the PSD (optimized) and base (baseline) checkpoints + build a few real seed clips."""
    import glob
    import sys

    import torch  # noqa: PLC0415

    sys.path.insert(0, f"{REMOTE_REPO}/scripts")
    from eval_world_model_offline import _build_loader, load_run_config  # noqa: PLC0415

    from mira.inference.loading import load_world_model  # noqa: PLC0415
    from mira.world_model.config import WorldModelInferenceConfig  # noqa: PLC0415

    def _local_codec(ckpt: str):
        hits = sorted(glob.glob(f"{Path(ckpt).parents[1]}/codec/**/checkpoint.pth", recursive=True))
        return hits[0] if hits else None

    opt_model, _ = load_world_model(Path(PSD_CKPT), device=device, codec_checkpoint=_local_codec(PSD_CKPT))
    base_model, _ = load_world_model(Path(BASE_CKPT), device=device, codec_checkpoint=_local_codec(BASE_CKPT))
    opt_model.eval()
    base_model.eval()
    opt_model.world_model.compile()
    opt_model.decode_to_video = torch.compile(opt_model.decode_to_video)

    opt_cfg = WorldModelInferenceConfig(n_diffusion_steps=2, schedule_type="linear_quadratic",
                                        noise_level=0.2, streaming_cache="ring", cuda_graphs=True)
    base_cfg = WorldModelInferenceConfig(n_diffusion_steps=10, schedule_type="linear_quadratic",
                                         noise_level=0.2, streaming_cache="grow", cuda_graphs=False)

    cfg = load_run_config(Path(PSD_CKPT))
    cfg.dataset.test_index = DATA_INDEX
    clip_len = (opt_model.n_context_latents + 2) * opt_model.temporal_downsampling
    loader = _build_loader(cfg, opt_model, clip_len=clip_len, batch_size=1, seed=7)
    it = iter(loader)
    seeds = []
    for _ in range(4):
        try:
            item = next(it)  # the loader yields (VideoActionBatch, metadata) tuples
            seeds.append(item[0] if isinstance(item, (tuple, list)) else item)
        except StopIteration:
            break

    models = {"optimized": (opt_model, opt_cfg), "baseline": (base_model, base_cfg)}
    return models, seeds


# --------------------------------------------------------------------------- ASGI app

OPT_SESSION = None   # optimized (2-step PSD + graphs)
BASE_SESSION = None  # released baseline (10-step, eager)


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=3600,
              min_containers=1, max_containers=1)
@modal.concurrent(max_inputs=1)  # shared CUDA-graph buffers -> one interactive rollout per container
@modal.asgi_app()
def web():
    import base64
    import io
    import json
    import traceback

    import torch
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from PIL import Image
    from starlette.concurrency import run_in_threadpool

    device = torch.device("cuda")
    print("Loading models (this warms both checkpoints)...", flush=True)
    models, seeds = _load_models(device)

    global OPT_SESSION, BASE_SESSION
    # Two sessions from the SAME seed clip: same keystrokes drive both, so it's the same scene, ours vs
    # released, side by side. (They're stepped sequentially per request -> the shared CUDA-graph buffers
    # are never used concurrently.)
    OPT_SESSION = LiveSession(models, seeds, device, mode="optimized")
    BASE_SESSION = LiveSession(models, seeds, device, mode="baseline")
    # Warm at startup so the first interaction is snappy (optimized compile + graph capture). Any bug in
    # step()/decode surfaces HERE, before the URL goes live -- read the logs above "Ready.".
    print("Warming (torch.compile + CUDA-graph capture, ~1-2 min)...", flush=True)
    for _ in range(2):
        OPT_SESSION.step([])
    BASE_SESSION.step([])
    print("Ready. URL is live; drive with WASD.", flush=True)

    from fastapi.exceptions import RequestValidationError

    fapp = FastAPI()

    # ---- verbose debug instrumentation -------------------------------------------------------------
    @fapp.middleware("http")
    async def _log_mw(request: Request, call_next):
        print(f"[MW] --> {request.method} {request.url.path} "
              f"ct={request.headers.get('content-type')!r} cl={request.headers.get('content-length')!r}",
              flush=True)
        try:
            resp = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            print(f"[MW] !! EXCEPTION in {request.url.path}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            raise
        clen = resp.headers.get("content-length")
        print(f"[MW] <-- {request.method} {request.url.path} -> {resp.status_code} "
              f"media={resp.media_type!r} clen={clen}", flush=True)
        return resp

    @fapp.exception_handler(RequestValidationError)
    async def _veh(request: Request, exc: RequestValidationError):
        raw = await request.body()
        print(f"[422] RequestValidationError {request.url.path}: {exc.errors()} raw={raw[:300]!r}",
              flush=True)
        return JSONResponse(status_code=422, content={"event": "error", "detail": str(exc.errors())})

    @fapp.exception_handler(Exception)
    async def _geh(request: Request, exc: Exception):
        print(f"[ERR] unhandled {request.url.path}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"event": "error", "detail": str(exc)})
    # ------------------------------------------------------------------------------------------------

    @fapp.get("/")
    def index():
        return HTMLResponse(HTML)

    @fapp.get("/ping")
    def ping():
        return JSONResponse({"ok": True, "ready": OPT_SESSION is not None})

    @fapp.post("/echo")
    def echo(request: Request):
        # No model, tiny fixed response -> isolates transport (does ANY POST round-trip work?).
        return JSONResponse({"ok": True, "echo": True})

    def _panel(session, keys: list[int], tag: str, want_frame: bool = True) -> dict:
        frame, ms = session.step(keys)
        b64 = ""
        if want_frame:
            img = Image.fromarray(frame)
            img.thumbnail((480, 480))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=72)
            b64 = base64.b64encode(buf.getvalue()).decode()
        ms = max(float(ms), 0.1)
        fps = 1000.0 / ms * session.td
        out = {
            "frame": b64, "ms": round(ms, 1), "fps": round(fps, 1),
            "rt": round(fps / float(session.model.config.video.fps), 2),
            "fpd": int(fps * 3600.0 / H100_USD_PER_HR),
            "steps": int(session.cfg.n_diffusion_steps),
        }
        print(f"[PANEL {tag}] frame={frame.shape} dtype={frame.dtype} jpeg_b64={len(b64)}B "
              f"ms={out['ms']} fps={out['fps']} types={[type(v).__name__ for v in out.values()]}",
              flush=True)
        return out

    def _process(body: dict) -> dict:
        """All the (blocking) GPU work; run in a threadpool so the event loop stays free."""
        try:
            cmd = body.get("cmd")
            print(f"[PROC] cmd={cmd!r} keys={body.get('keys')!r}", flush=True)
            if cmd == "reset":
                OPT_SESSION.reset(); BASE_SESSION.reset()
                return {"event": "reset"}
            if cmd == "seed":
                OPT_SESSION.next_seed(); BASE_SESSION.next_seed()
                return {"event": "seed", "idx": OPT_SESSION.seed_idx}
            keys = body.get("keys", [])
            want_frame = not body.get("noframe")  # POST {"noframe":true} -> metrics only (isolate payload)
            return {"opt": _panel(OPT_SESSION, keys, "opt", want_frame),
                    "base": _panel(BASE_SESSION, keys, "base", want_frame)}
        except Exception as exc:  # noqa: BLE001 -- surface errors to the client for live debugging
            traceback.print_exc()
            return {"event": "error", "detail": f"{type(exc).__name__}: {exc}"}

    @fapp.post("/step")
    async def step(request: Request):
        raw = await request.body()
        print(f"[STEP] raw_len={len(raw)} raw_head={raw[:120]!r}", flush=True)
        try:
            body = json.loads(raw) if raw else {}
        except Exception as exc:  # noqa: BLE001
            print(f"[STEP] body parse failed: {exc}", flush=True)
            body = {}
        result = await run_in_threadpool(_process, body)
        try:
            payload = json.dumps(result, allow_nan=False)
        except (ValueError, TypeError) as exc:
            print(f"[STEP] serialize failed: {exc}", flush=True)
            payload = json.dumps({"event": "error", "detail": f"serialize: {exc}"})
        print(f"[STEP] returning {len(payload)}B, keys={list(result.keys())}", flush=True)
        return JSONResponse(content=json.loads(payload))

    return fapp


# --------------------------------------------------------------------------- browser UI

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIRA-Mini - live</title>
<style>
  :root { color-scheme: dark; --bg:#0a0c0f; --panel:#0f1318; --line:#20262d; --dim:#7c8894; --fg:#eef3f7;
          --green:#43e296; --amber:#f0a55e; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); padding:24px 16px 40px;
         font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif;
         display:flex; flex-direction:column; align-items:center; gap:18px; }
  header { text-align:center; }
  h1 { font-size:20px; margin:0 0 4px; font-weight:650; letter-spacing:-.01em; }
  .sub { color:var(--dim); margin:0; font-size:13px; }
  .panels { display:flex; gap:20px; width:100%; max-width:1080px; justify-content:center; flex-wrap:wrap; }
  .panel { flex:1 1 460px; max-width:520px; }
  .view { position:relative; aspect-ratio:16/9; background:#000; border:1px solid var(--line);
          border-radius:12px; overflow:hidden; }
  .view.opt  { border-color:#1d5b45; box-shadow:0 0 0 1px #12281f, 0 8px 30px -12px #0d3b2a; }
  .view.base { border-color:#5b3a1d; box-shadow:0 0 0 1px #281c12, 0 8px 30px -12px #3b2a0d; }
  canvas { width:100%; height:100%; display:block; }
  .label { display:flex; align-items:center; gap:8px; margin:12px 2px 8px; font-weight:650; font-size:15px; }
  .dot { width:9px; height:9px; border-radius:50%; }
  .opt .dot, .l-opt .dot { background:var(--green); box-shadow:0 0 8px var(--green); }
  .base .dot, .l-base .dot { background:var(--amber); box-shadow:0 0 8px var(--amber); }
  .label small { color:var(--dim); font-weight:500; font-size:12px; }
  .metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--line);
             border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .metric { background:var(--panel); padding:10px 12px; }
  .metric .n { font-size:20px; font-weight:680; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
  .l-opt .metric .n { color:var(--green); } .l-base .metric .n { color:var(--amber); }
  .metric .u { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.04em; margin-top:2px; }
  .summary { font-size:15px; color:var(--dim); text-align:center; min-height:22px; }
  .summary b { color:var(--fg); font-weight:680; }
  .summary .hi { color:var(--green); }
  .controls { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; }
  button { background:var(--panel); color:var(--fg); border:1px solid var(--line); border-radius:9px;
           padding:9px 16px; font:inherit; cursor:pointer; transition:border-color .15s; }
  button:hover { border-color:#3a444e; }
  .keys { color:var(--dim); font-size:12px; text-align:center; }
  kbd { background:var(--panel); border:1px solid var(--line); border-bottom-width:2px; border-radius:4px;
        padding:1px 6px; color:#cbd5dd; font-size:11px; }
  #status { color:var(--dim); font-size:12px; min-height:16px; }
</style></head>
<body>
  <header>
    <h1>MIRA-Mini &middot; live world model</h1>
    <p class="sub">Same scene, same controls. Our optimized inference vs the released baseline &mdash; measured live on one H100.</p>
  </header>

  <div class="panels">
    <div class="panel l-opt">
      <div class="view opt"><canvas id="cv-opt" width="640" height="360"></canvas></div>
      <div class="label opt"><span class="dot"></span>Ours <small>&mdash; 2-step PSD + torch.compile + CUDA graphs</small></div>
      <div class="metrics">
        <div class="metric"><div class="n" id="opt-fps">&ndash;</div><div class="u">gen fps</div></div>
        <div class="metric"><div class="n" id="opt-ms">&ndash;</div><div class="u">ms / frame</div></div>
        <div class="metric"><div class="n" id="opt-rt">&ndash;</div><div class="u">&times; realtime</div></div>
        <div class="metric"><div class="n" id="opt-fpd">&ndash;</div><div class="u">frames / $</div></div>
      </div>
    </div>
    <div class="panel l-base">
      <div class="view base"><canvas id="cv-base" width="640" height="360"></canvas></div>
      <div class="label base"><span class="dot"></span>Released <small>&mdash; 10-step base, eager</small></div>
      <div class="metrics">
        <div class="metric"><div class="n" id="base-fps">&ndash;</div><div class="u">gen fps</div></div>
        <div class="metric"><div class="n" id="base-ms">&ndash;</div><div class="u">ms / frame</div></div>
        <div class="metric"><div class="n" id="base-rt">&ndash;</div><div class="u">&times; realtime</div></div>
        <div class="metric"><div class="n" id="base-fpd">&ndash;</div><div class="u">frames / $</div></div>
      </div>
    </div>
  </div>

  <div class="summary" id="summary">warming up&hellip;</div>
  <div class="controls">
    <button id="btn-reset">Reset world</button>
    <button id="btn-seed">New scene</button>
  </div>
  <div class="keys">
    <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> drive &middot; <kbd>Q</kbd><kbd>E</kbd> air-roll &middot;
    <kbd>Space</kbd> ball-cam &middot; <kbd>Shift</kbd> boost &middot; <kbd>Ctrl</kbd>
  </div>
  <div id="status">connecting&hellip;</div>

<script>
// Key vocab order must match DEFAULT_RL_KEYS in mira/data/actions.py.
const VOCAB = ["W","A","S","D","Q","E","Space","LShiftKey","LControlKey"];
const CODE2KEY = {KeyW:"W",KeyA:"A",KeyS:"S",KeyD:"D",KeyQ:"Q",KeyE:"E",
                  Space:"Space",ShiftLeft:"LShiftKey",ShiftRight:"LShiftKey",
                  ControlLeft:"LControlKey",ControlRight:"LControlKey"};
const held = new Set();
const $ = id => document.getElementById(id);
const ctx = {opt: $("cv-opt").getContext("2d"), base: $("cv-base").getContext("2d")};
const imgs = {opt: new Image(), base: new Image()};
imgs.opt.onload  = () => ctx.opt.drawImage(imgs.opt, 0, 0, 640, 360);
imgs.base.onload = () => ctx.base.drawImage(imgs.base, 0, 0, 640, 360);
let running = true;
const sleep = ms => new Promise(r => setTimeout(r, ms));

function keyIdxs(){ return [...held].map(k => VOCAB.indexOf(k)).filter(i => i>=0); }
addEventListener("keydown", e => { const k=CODE2KEY[e.code]; if(k){ held.add(k); e.preventDefault(); }});
addEventListener("keyup",   e => { const k=CODE2KEY[e.code]; if(k){ held.delete(k); e.preventDefault(); }});

async function post(body){
  const r = await fetch("/step", {method:"POST", headers:{"Content-Type":"application/json"},
                                  body: JSON.stringify(body)});
  return r.json();
}
function paint(which, m){
  imgs[which].src = "data:image/jpeg;base64," + m.frame;
  $(which+"-fps").textContent = m.fps.toFixed(1);
  $(which+"-ms").textContent  = m.ms.toFixed(1);
  $(which+"-rt").textContent  = m.rt.toFixed(2) + "x";
  $(which+"-fpd").textContent = m.fpd >= 1000 ? Math.round(m.fpd/1000) + "k" : m.fpd;
}
async function loop(){
  $("status").textContent = "live - click the page and drive with WASD";
  while(running){
    try {
      const r = await post({keys: keyIdxs()});
      if(r.event === "error"){ $("status").textContent = "server error: " + r.detail; await sleep(1000); continue; }
      if(r.opt && r.base){
        paint("opt", r.opt); paint("base", r.base);
        const sx = (r.opt.fps / r.base.fps).toFixed(1), cx = (r.opt.fpd / r.base.fpd).toFixed(1);
        $("summary").innerHTML = "Ours generates <b class='hi'>" + sx + "&times; more frames per second</b> and <b class='hi'>"
          + cx + "&times; more frames per dollar</b> &mdash; " + r.opt.steps + " diffusion steps vs " + r.base.steps + ".";
      }
    } catch(e){ $("status").textContent = "network error: " + e; await sleep(500); }
  }
}
$("btn-reset").onclick = () => post({cmd:"reset"});
$("btn-seed").onclick  = () => post({cmd:"seed"});
loop();
</script>
</body></html>
"""
