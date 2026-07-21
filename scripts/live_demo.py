"""Live, interactive world-model demo: drive MIRA-Mini with the keyboard and watch it generate in
real time, toggling between our optimized inference (2-step PSD + torch.compile + CUDA graphs) and the
released baseline (10-step base, eager) — with a live, server-measured speed/cost HUD.

It is legit end-to-end: a real H100 holds both real checkpoints warm, bootstraps from a real
rocket-science clip, and streams actual generated frames as you press keys. The HUD numbers are
measured on the GPU per frame (torch.cuda.Event), not scripted. Transport is HTTP polling (one POST
per frame) — robust on Modal's runtime; the per-frame GPU time is what the HUD reports.

This is now a thin **reference client over MiraEngine** (the sibling mira-engine package): the demo only
calls `set_context` / `gen_frame` / `render`; the engine owns model loading, the fast stack, and the
streaming state — mirroring how world_engine's examples drive `WorldEngine`.

    modal serve scripts/live_demo.py     # dev: prints a *.modal.run URL, open it, drive with WASD
    modal deploy scripts/live_demo.py    # persistent URL (run `modal app stop mira-live-demo` when done)

Controls: W/A/S/D drive, Q/E air-roll, Space ball-cam, Shift boost/powerslide, Ctrl. Buttons toggle
Ours vs Released and reset the world. See docs/optimization_plan.md (E2) for the ~7x result.

NOTE: do NOT add `from __future__ import annotations` here. It stringifies annotations (PEP 563), and
FastAPI then resolves `request: Request` against the *module* globals -- but Request is imported inside
web(), so it can't resolve it and 422s the param as a missing query field.
"""

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
    # The MiraEngine package (sibling repo) -- the demo is now a thin reference client over it.
    .add_local_dir(str(REPO_ROOT.parent / "mira-engine"), "/root/mira-engine", copy=True,
                   ignore=["**/.git", "**/__pycache__", "**/*.mp4", "**/*.egg-info"])
    .run_commands("pip install -e /root/mira-engine --no-deps")
    .run_function(_prefetch_dino)
)

data_vol = modal.Volume.from_name("mira-bench-data", create_if_missing=True)
app = modal.App("mira-live-demo", image=image)


# --------------------------------------------------------------------------- engines (reference client)


def _load_engines(device):
    """Build the optimized (2-step PSD) + baseline (10-step base) MiraEngine instances + seed clips.

    The demo is a thin client over MiraEngine -- the engine owns model loading, the fast stack, and the
    streaming state; the demo just calls set_context / gen_frame / render.
    """
    import sys

    from mira_engine import MiraEngine  # noqa: PLC0415

    opt = MiraEngine(PSD_CKPT, device=device, n_diffusion_steps=2, schedule_type="linear_quadratic",
                     noise_level=0.2, compile=True, cuda_graphs=True)
    base = MiraEngine(BASE_CKPT, device=device, n_diffusion_steps=10, schedule_type="linear_quadratic",
                      noise_level=0.2, compile=True, cuda_graphs=False)

    # Seed clips from the real test split, reusing the (already-loaded) optimized engine's model.
    sys.path.insert(0, f"{REMOTE_REPO}/scripts")
    from eval_world_model_offline import _build_loader, load_run_config  # noqa: PLC0415

    cfg = load_run_config(Path(PSD_CKPT))
    cfg.dataset.test_index = DATA_INDEX
    m = opt.model
    clip_len = (m.n_context_latents + 2) * m.temporal_downsampling
    loader = _build_loader(cfg, m, clip_len=clip_len, batch_size=1, seed=7)
    it = iter(loader)
    seeds = []
    for _ in range(4):
        try:
            item = next(it)  # the loader yields (VideoActionBatch, metadata) tuples
            seeds.append(item[0] if isinstance(item, (tuple, list)) else item)
        except StopIteration:
            break
    return {"optimized": opt, "baseline": base}, seeds


# --------------------------------------------------------------------------- ASGI app

OPT_ENGINE = None    # MiraEngine: optimized (2-step PSD + graphs)
BASE_ENGINE = None   # MiraEngine: released baseline (10-step, eager)
SEEDS: list = []
SEED_IDX = 0


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
    from mira_engine import CtrlInput
    from PIL import Image
    from starlette.concurrency import run_in_threadpool

    device = torch.device("cuda")
    print("Loading models (this warms both checkpoints)...", flush=True)
    engines, seeds = _load_engines(device)

    global OPT_ENGINE, BASE_ENGINE, SEEDS, SEED_IDX
    OPT_ENGINE, BASE_ENGINE, SEEDS, SEED_IDX = engines["optimized"], engines["baseline"], seeds, 0
    # Same seed clip -> same scene, ours vs released. set_context bootstraps each engine's world.
    OPT_ENGINE.set_context(SEEDS[0])
    BASE_ENGINE.set_context(SEEDS[0])
    print("Warming (torch.compile + CUDA-graph capture, ~1-2 min)...", flush=True)
    for _ in range(2):
        OPT_ENGINE.gen_frame(CtrlInput(), return_img=False); OPT_ENGINE.render()
    BASE_ENGINE.gen_frame(CtrlInput(), return_img=False); BASE_ENGINE.render()
    OPT_ENGINE.set_context(SEEDS[0]); BASE_ENGINE.set_context(SEEDS[0])  # fresh start after warmup
    print("Ready. URL is live; drive with WASD.", flush=True)

    fapp = FastAPI()

    @fapp.exception_handler(Exception)
    async def _on_error(request: Request, exc: Exception):
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"event": "error", "detail": str(exc)})

    @fapp.get("/")
    def index():
        return HTMLResponse(HTML)

    @fapp.get("/ping")
    def ping():
        return JSONResponse({"ok": True, "ready": OPT_ENGINE is not None})

    def _panel(engine, keys: list[int]) -> dict:
        # Time the diffusion only (the thing we optimized: 2-step vs 10-step); decode separately for
        # display, so the Ours-vs-Released number isn't diluted by the shared, constant codec cost.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        engine.gen_frame(CtrlInput(button=set(keys)), return_img=False)
        end.record()
        torch.cuda.synchronize()
        ms = max(start.elapsed_time(end), 0.1)
        img = engine.render()[-1]  # (H, W, 3) uint8, newest frame
        buf = io.BytesIO()
        Image.fromarray(img.cpu().numpy()).save(buf, format="JPEG", quality=90)
        fps = 1000.0 / ms * engine._td
        return {
            "frame": base64.b64encode(buf.getvalue()).decode(),
            "ms": round(ms, 1), "fps": round(fps, 1),
            "rt": round(fps / float(engine.model.config.video.fps), 2),
            "fpd": int(fps * 3600.0 / H100_USD_PER_HR),
            "steps": int(engine.n_diffusion_steps),
        }

    def _process(body: dict) -> dict:
        """All the (blocking) GPU work; run in a threadpool so the event loop stays free."""
        global SEED_IDX
        try:
            cmd = body.get("cmd")
            if cmd == "reset":
                OPT_ENGINE.set_context(SEEDS[SEED_IDX]); BASE_ENGINE.set_context(SEEDS[SEED_IDX])
                return {"event": "reset"}
            if cmd == "seed":
                SEED_IDX = (SEED_IDX + 1) % len(SEEDS)
                OPT_ENGINE.set_context(SEEDS[SEED_IDX]); BASE_ENGINE.set_context(SEEDS[SEED_IDX])
                return {"event": "seed", "idx": SEED_IDX}
            keys = body.get("keys", [])
            which = body.get("which")  # client polls each panel on its own loop (one at a time)
            if which == "opt":
                return {"opt": _panel(OPT_ENGINE, keys)}
            if which == "base":
                return {"base": _panel(BASE_ENGINE, keys)}
            return {"opt": _panel(OPT_ENGINE, keys), "base": _panel(BASE_ENGINE, keys)}
        except Exception as exc:  # noqa: BLE001 -- surface errors to the client instead of a blank frame
            traceback.print_exc()
            return {"event": "error", "detail": f"{type(exc).__name__}: {exc}"}

    # NB: do NOT type the body param with a Pydantic model here and keep `request: Request` -- see the
    # module docstring on why `from __future__ import annotations` must stay out of this file.
    @fapp.post("/step")
    async def step(request: Request):
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except Exception:  # noqa: BLE001
            body = {}
        result = await run_in_threadpool(_process, body)
        return JSONResponse(content=result)

    return fapp


# --------------------------------------------------------------------------- browser UI

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>live</title>
<style>
  * { box-sizing:border-box; }
  body { margin:0; background:#fff; color:#000; padding:28px 16px 44px;
         font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif;
         display:flex; flex-direction:column; align-items:center; gap:20px; }
  .panels { display:flex; gap:20px; width:100%; max-width:1120px; justify-content:center; flex-wrap:wrap; }
  .panel { flex:1 1 480px; max-width:540px; }
  .view { position:relative; aspect-ratio:16/9; background:#000; border:1px solid #000;
          border-radius:12px; overflow:hidden; }
  canvas { width:100%; height:100%; display:block; }
  .overlay { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
             background:rgba(255,255,255,.66); cursor:pointer; }
  .overlay.hidden { display:none; }
  .play { font:inherit; font-size:15px; font-weight:650; padding:12px 22px; border-radius:10px;
          background:#fff; border:1px solid #000; color:#000; cursor:pointer; }
  .play:hover { background:#000; color:#fff; }
  .summary { font-size:15px; color:#000; text-align:center; min-height:22px; }
  .summary b { font-weight:750; }
  .controls { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; }
  button.ctrl { background:#fff; color:#000; border:1px solid #000; border-radius:9px;
                padding:9px 16px; font:inherit; cursor:pointer; }
  button.ctrl:hover { background:#000; color:#fff; }
  .keys { color:#555; font-size:12px; text-align:center; }
  kbd { background:#f2f2f2; border:1px solid #bbb; border-bottom-width:2px; border-radius:4px;
        padding:1px 6px; color:#000; font-size:11px; }
</style></head>
<body>
  <div class="panels">
    <div class="panel">
      <div class="view">
        <canvas id="cv-opt" width="640" height="360"></canvas>
        <div class="overlay" id="ov-opt" data-which="opt"><button class="play">&#9654; Play &mdash; Ours (2-step)</button></div>
      </div>
    </div>
    <div class="panel">
      <div class="view">
        <canvas id="cv-base" width="640" height="360"></canvas>
        <div class="overlay" id="ov-base" data-which="base"><button class="play">&#9654; Play &mdash; Released (10-step)</button></div>
      </div>
    </div>
  </div>
  <div class="summary" id="summary"></div>
  <div class="controls">
    <button class="ctrl" id="btn-reset">Reset world</button>
    <button class="ctrl" id="btn-seed">New scene</button>
  </div>
  <div class="keys">
    <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> drive &middot; <kbd>Q</kbd><kbd>E</kbd> air-roll &middot;
    <kbd>Space</kbd> ball-cam &middot; <kbd>Shift</kbd> boost &middot; <kbd>Ctrl</kbd>
  </div>
<script>
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
const best = {opt:null, base:null};
function updateSummary(){
  const o = best.opt, b = best.base;
  if(o && b){
    $("summary").innerHTML = "Ours: <b>" + (o.fps/b.fps).toFixed(1) + "&times; more frames / second</b> &middot; <b>"
      + (o.fpd/b.fpd).toFixed(1) + "&times; more frames / dollar</b> &middot; " + o.steps + " steps vs " + b.steps;
  } else {
    $("summary").textContent = "Play each side to compare — only one runs at a time so it gets the whole GPU.";
  }
}
function paint(which, m){
  imgs[which].src = "data:image/jpeg;base64," + m.frame;
  best[which] = m;
  updateSummary();
}
let active = null, loopId = 0;
async function play(which){
  const myId = ++loopId;
  active = which;
  ["opt","base"].forEach(w => $("ov-"+w).classList.toggle("hidden", w===which));
  while(running && loopId === myId){
    try {
      const r = await post({which, keys: keyIdxs()});
      if(r && r[which] && r[which].frame) paint(which, r[which]);
      else await sleep(400);
    } catch(e){ await sleep(400); }
  }
}
document.querySelectorAll(".overlay").forEach(ov => ov.onclick = () => play(ov.getAttribute("data-which")));
$("btn-reset").onclick = () => post({cmd:"reset"});
$("btn-seed").onclick  = () => post({cmd:"seed"});
updateSummary();
</script>
</body></html>
"""
