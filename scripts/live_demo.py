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

    def __init__(self, models: dict, seed_batches: list, device):
        import torch  # noqa: PLC0415

        self.torch = torch
        self.models = models          # {"optimized": (model, cfg), "baseline": (model, cfg)}
        self.seed_batches = seed_batches
        self.device = device
        self.seed_idx = 0
        self.set_mode("optimized")

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
        """Denoise + decode one latent from the held keys; return (uint8 HWC frames, gpu_ms)."""
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
            video = self.model.decode_to_video(self.z[:, -1:])  # (1, td, C, H, W) in [-1, 1]
            end.record()
        torch.cuda.synchronize()
        gpu_ms = start.elapsed_time(end)

        # [-1,1] -> uint8 HWC. Keep only the newest video frame for display.
        vid = ((video[0, -1].clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8)  # (C, H, W)
        frame = vid.permute(1, 2, 0).contiguous().cpu().numpy()                # (H, W, C)
        self.frame_idx += 1
        # Trim the action history so it can't grow unbounded over a long session.
        keep = (self.window + 2) * self.td
        if self.actions.key_presses.shape[1] > keep:
            self.actions = self.actions.slice_time(-keep, None)
        return frame, gpu_ms


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
            seeds.append(next(it))
        except StopIteration:
            break

    models = {"optimized": (opt_model, opt_cfg), "baseline": (base_model, base_cfg)}
    return models, seeds


# --------------------------------------------------------------------------- ASGI app

SESSION = None  # one interactive session per (single) container


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=3600,
              min_containers=1, max_containers=1)
@modal.concurrent(max_inputs=1)  # shared CUDA-graph buffers -> one interactive rollout per container
@modal.asgi_app()
def web():
    import base64
    import io
    import traceback

    import torch
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from PIL import Image
    from pydantic import BaseModel

    device = torch.device("cuda")
    print("Loading models (this warms both checkpoints)...", flush=True)
    models, seeds = _load_models(device)

    global SESSION
    SESSION = LiveSession(models, seeds, device)
    # Warm at startup so the first user interaction is snappy: this triggers the optimized-path compile
    # + graph capture and the baseline eager path. Any bug in step()/decode surfaces HERE, before the
    # URL goes live -- read the logs above "Ready:".
    print("Warming optimized path (torch.compile + CUDA-graph capture, ~1-2 min)...", flush=True)
    for _ in range(2):
        SESSION.step([])
    SESSION.set_mode("baseline")
    SESSION.step([])
    SESSION.set_mode("optimized")
    print("Ready. URL is live; drive with WASD.", flush=True)

    fapp = FastAPI()

    class StepReq(BaseModel):
        keys: list[int] = []
        cmd: str | None = None
        mode: str | None = None

    @fapp.get("/")
    def index():
        return HTMLResponse(HTML)

    # Sync endpoint -> FastAPI runs it in a threadpool, so the GPU step never blocks the event loop.
    @fapp.post("/step")
    def step(req: StepReq):
        try:
            if req.cmd == "mode":
                SESSION.set_mode(req.mode or "optimized")
                SESSION.step([])  # recapture graph for the newly selected mode
                return {"event": "mode", "mode": SESSION.mode}
            if req.cmd == "reset":
                SESSION.reset()
                return {"event": "reset"}
            if req.cmd == "seed":
                SESSION.next_seed()
                return {"event": "seed", "idx": SESSION.seed_idx}

            frame, gpu_ms = SESSION.step(req.keys)
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()

            gen_fps = 1000.0 / gpu_ms * SESSION.td
            x_realtime = gen_fps / SESSION.model.config.video.fps
            frames_per_usd = gen_fps * 3600.0 / H100_USD_PER_HR
            return {
                "frame": b64, "mode": SESSION.mode,
                "gpu_ms": round(gpu_ms, 1), "gen_fps": round(gen_fps, 1),
                "x_realtime": round(x_realtime, 2), "frames_per_usd": int(frames_per_usd),
                "n_steps": SESSION.cfg.n_diffusion_steps,
            }
        except Exception as exc:  # noqa: BLE001 -- surface errors to the client for live debugging
            traceback.print_exc()
            return {"event": "error", "detail": f"{type(exc).__name__}: {exc}"}

    return fapp


# --------------------------------------------------------------------------- browser UI

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>MIRA-Mini - live</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0d10; color:#e8eef2; font:14px/1.4 ui-sans-serif,system-ui,sans-serif;
         display:flex; flex-direction:column; align-items:center; gap:16px; padding:20px; }
  h1 { font-size:18px; margin:4px 0; font-weight:600; }
  .sub { color:#8a97a3; margin:0 0 6px; }
  #stage { position:relative; width:640px; max-width:96vw; aspect-ratio:16/9; background:#000;
           border:1px solid #22282e; border-radius:10px; overflow:hidden; }
  canvas { width:100%; height:100%; display:block; }
  #hud { position:absolute; top:10px; left:10px; display:grid; grid-template-columns:auto auto; gap:2px 14px;
         background:rgba(8,10,13,.72); padding:10px 14px; border-radius:8px; font-variant-numeric:tabular-nums;
         backdrop-filter:blur(4px); }
  #hud .k { color:#8a97a3; } #hud .v { text-align:right; font-weight:600; }
  #hud .big { font-size:18px; }
  .badge { position:absolute; top:10px; right:10px; padding:4px 10px; border-radius:999px; font-weight:600;
           font-size:12px; }
  .opt { background:#0e3b2e; color:#4ade9b; border:1px solid #1c6b52; }
  .base { background:#3b1e0e; color:#f0a662; border:1px solid #6b451c; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:center; }
  button { background:#161b21; color:#e8eef2; border:1px solid #2a323a; border-radius:8px; padding:8px 14px;
           font:inherit; cursor:pointer; }
  button:hover { border-color:#3a444e; } button.on { background:#12351f; border-color:#1c6b52; color:#4ade9b; }
  .keys { color:#8a97a3; font-size:12px; }
  kbd { background:#161b21; border:1px solid #2a323a; border-bottom-width:2px; border-radius:4px;
        padding:1px 6px; color:#cdd7df; }
  #status { color:#8a97a3; min-height:18px; }
  .cmp { color:#8a97a3; font-size:12px; } .cmp b { color:#e8eef2; }
</style></head>
<body>
  <h1>MIRA-Mini &middot; live world model</h1>
  <p class="sub">Drive it. Toggle our optimized inference vs the released baseline. Metrics measured on the H100.</p>
  <div id="stage">
    <canvas id="cv" width="640" height="360"></canvas>
    <div id="badge" class="badge opt">OURS &middot; 2-step PSD + graphs</div>
    <div id="hud">
      <div class="k big">gen fps</div><div class="v big" id="fps">-</div>
      <div class="k">ms / frame</div><div class="v" id="ms">-</div>
      <div class="k">x realtime</div><div class="v" id="rt">-</div>
      <div class="k">frames / $</div><div class="v" id="fpd">-</div>
      <div class="k">diffusion steps</div><div class="v" id="steps">-</div>
    </div>
  </div>
  <div class="row">
    <button id="btn-opt" class="on">Ours (2-step PSD + graphs)</button>
    <button id="btn-base">Released (10-step, eager)</button>
    <button id="btn-reset">Reset world</button>
    <button id="btn-seed">New scene</button>
  </div>
  <div class="cmp" id="cmp"></div>
  <div class="keys">
    <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> drive &middot;
    <kbd>Q</kbd><kbd>E</kbd> air-roll &middot; <kbd>Space</kbd> ball-cam &middot; <kbd>Shift</kbd> boost &middot; <kbd>Ctrl</kbd>
  </div>
  <div id="status">connecting...</div>
<script>
// Key vocab order must match DEFAULT_RL_KEYS in mira/data/actions.py.
const VOCAB = ["W","A","S","D","Q","E","Space","LShiftKey","LControlKey"];
const CODE2KEY = {KeyW:"W",KeyA:"A",KeyS:"S",KeyD:"D",KeyQ:"Q",KeyE:"E",
                  Space:"Space",ShiftLeft:"LShiftKey",ShiftRight:"LShiftKey",
                  ControlLeft:"LControlKey",ControlRight:"LControlKey"};
const held = new Set();
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const img = new Image();
const best = {opt:null, base:null};
const $ = id => document.getElementById(id);
let mode = "optimized", running = true;

function keyIdxs(){ return [...held].map(k => VOCAB.indexOf(k)).filter(i => i>=0); }
addEventListener("keydown", e => { const k=CODE2KEY[e.code]; if(k){ held.add(k); e.preventDefault(); }});
addEventListener("keyup",   e => { const k=CODE2KEY[e.code]; if(k){ held.delete(k); e.preventDefault(); }});

async function post(body){
  const r = await fetch("/step", {method:"POST", headers:{"Content-Type":"application/json"},
                                  body: JSON.stringify(body)});
  return r.json();
}
function draw(m){
  img.onload = () => ctx.drawImage(img, 0, 0, cv.width, cv.height);
  img.src = "data:image/jpeg;base64," + m.frame;
  $("fps").textContent = m.gen_fps.toFixed(1);
  $("ms").textContent  = m.gpu_ms.toFixed(1) + " ms";
  $("rt").textContent  = m.x_realtime.toFixed(2) + "x";
  $("fpd").textContent = m.frames_per_usd.toLocaleString();
  $("steps").textContent = m.n_steps;
  best[mode==="optimized"?"opt":"base"] = m;
  const o = best.opt, b = best.base;
  if(o && b){
    $("cmp").innerHTML = "Ours vs Released: <b>" + (o.gen_fps/b.gen_fps).toFixed(1) + "x faster</b> ("
      + o.gen_fps + " vs " + b.gen_fps + " fps) &middot; <b>"
      + (o.frames_per_usd/b.frames_per_usd).toFixed(1) + "x more frames per dollar</b>";
  }
}
async function loop(){
  $("status").textContent = "connected - click the page and drive with WASD";
  while(running){
    try {
      const m = await post({keys: keyIdxs()});
      if(m.event === "error"){ $("status").textContent = "server error: " + m.detail; await sleep(1000); }
      else if(m.frame){ draw(m); }
    } catch(e){ $("status").textContent = "network error: " + e; await sleep(500); }
  }
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function setMode(m){
  mode = m;
  await post({cmd:"mode", mode:m});
  const opt = m === "optimized";
  $("btn-opt").classList.toggle("on", opt); $("btn-base").classList.toggle("on", !opt);
  const badge = $("badge");
  badge.className = "badge " + (opt ? "opt" : "base");
  badge.textContent = opt ? "OURS - 2-step PSD + graphs" : "RELEASED - 10-step, eager";
}
$("btn-opt").onclick  = () => setMode("optimized");
$("btn-base").onclick = () => setMode("baseline");
$("btn-reset").onclick= () => post({cmd:"reset"});
$("btn-seed").onclick = () => post({cmd:"seed"});
loop();
</script>
</body></html>
"""
