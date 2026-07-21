"""Modal entrypoints to run the mira training/inference benchmarks on H100s.

Mirrors the conventions in ``lewm-modal-fast/modal_app.py`` (Image -> Volume -> H100 functions ->
local_entrypoints). The whole repo is baked into the image and installed editable; the DINOv3
backbone code is pre-fetched at build time (``pretrained=False`` -- code only, no gated weights) so
the random-init codec constructs without a runtime torch.hub round-trip.

Every benchmark runs as a subprocess of the repo's own CLI (scripts/bench_*.py) so the local and
remote code paths are identical and DDP works via torchrun.

Quick start (see docs/benchmarking.md for the full matrix and how to read the numbers):
    modal run scripts/modal_bench.py::infer
    modal run scripts/modal_bench.py::train --mode dit --compile
    modal run scripts/modal_bench.py::ablation
    modal run scripts/modal_bench.py::ddp
    modal run scripts/modal_bench.py::profile --mode dit
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/mira"
SCRIPTS_DIR = f"{REMOTE_REPO}/scripts"


def _prefetch_dino() -> None:
    """Cache the DINOv3 hub repo + architecture at image-build time (no gated weights)."""
    import torch

    # trust_repo + skip_validation avoid torch.hub's unauthenticated GitHub API fork-validation call
    # (60 req/hr -> 403 "rate limit exceeded" on rebuilds); the zipball still downloads and caches.
    torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model="dinov3_vitl16",
        source="github",
        verbose=False,
        pretrained=False,
        trust_repo=True,
        skip_validation=True,
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "ffmpeg")
    # CUDA (cu128) torch + torchvision first, then CPU-decode torchcodec (its ABI must match torch;
    # --no-deps keeps the cu128 torch build). Mirrors the repo's pixi setup. torchvision is installed
    # here from the cu128 index so the later `[train]` -> lpips dependency is satisfied by an
    # ABI-matched build rather than pulling a default-PyPI torchvision that mismatches cu128 torch.
    .pip_install(
        "torch==2.8.*", "torchvision==0.23.*", extra_index_url="https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "pip install --no-deps torchcodec==0.7.0 --index-url https://download.pytorch.org/whl/cpu"
    )
    .add_local_dir(
        str(REPO_ROOT),
        REMOTE_REPO,
        copy=True,
        ignore=["**/.git", "**/.pixi", "**/__pycache__", "**/*.lock", "**/.pytest_cache", "**/.benchmarks"],
    )
    .run_commands(f"pip install -e '{REMOTE_REPO}[train]'")
    # FDD metric dep for the quality gate (scipy already comes via lpips). Not adding mira[eval]
    # wholesale because that would reinstall torchcodec from the default index over the cpu build.
    .pip_install("pytorch-fid")
    # E2 Phase 1: pull the mira-mini / mira-mini-psd checkpoints and the rocket-science split from the
    # Hub onto the volume (huggingface_hub is only an optional [hf] extra of the repo, so install it here).
    .pip_install("huggingface_hub")
    # Category C: torchao weight-only int8/fp8 quantization. Pinned to a torch-2.8-era release so its
    # CUDA kernels match (the latest torchao targets torch>=2.11 and silently skips its cpp extensions,
    # which would make int8 fall back to a slow path). --no-deps keeps the cu128 torch build.
    .run_commands("pip install --no-deps torchao==0.12.0")
    .run_function(_prefetch_dino)
)

# Phase 44: the sibling MiraEngine repo layered onto the bench image (which already has mira + torch +
# torchao), installed editable --no-deps, so we can verify the engine end to end on the staged checkpoint.
MIRA_ENGINE_DIR = REPO_ROOT.parent / "mira-engine"
engine_image = image.add_local_dir(
    str(MIRA_ENGINE_DIR), "/root/mira-engine", copy=True,
    ignore=["**/.git", "**/__pycache__", "**/*.mp4", "**/*.egg-info"],
).run_commands("pip install -e /root/mira-engine --no-deps")

data_vol = modal.Volume.from_name("mira-bench-data", create_if_missing=True)

app = modal.App("mira-bench", image=image)


def _run(cmd: list[str], nproc: int | None = None, visible: str | None = None,
         extra_env: dict | None = None) -> str:
    """Run a bench CLI as a subprocess in the scripts dir; optionally under torchrun for DDP."""
    import os

    env = os.environ.copy()
    if extra_env:
        env.update({k: v for k, v in extra_env.items() if v is not None})
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    if nproc and nproc > 1:
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}"] + cmd
    else:
        cmd = ["python"] + cmd
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=SCRIPTS_DIR, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark subprocess failed ({proc.returncode})")
    return "ok"


# --------------------------------------------------------------------------- inference


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=3600)
def infer(
    n_diffusion_steps: str = "1 2 4 8",
    n_frames: int = 16,
    compile: bool = False,
    tf32: bool = False,
    batch: int = 1,
    optim: str = "",
    streaming_cache: str = "grow",
    compile_mode: str = "default",
    quantize: str = "none",
    cuda_graphs: bool = False,
    verify_graphs: bool = False,
    psd: bool = False,
) -> None:
    """Rollout latency + decode throughput + end-to-end, over the diffusion-step sweep."""
    cmd = [
        "bench_infer_speed.py",
        "--n-diffusion-steps", *n_diffusion_steps.split(),
        "--n-frames", str(n_frames),
        "--batch", str(batch),
        "--streaming-cache", streaming_cache,
        "--compile-mode", compile_mode,
        "--quantize", quantize,
    ]
    if compile:
        cmd.append("--compile")
    if cuda_graphs:
        cmd.append("--cuda-graphs")
    if verify_graphs:
        cmd.append("--verify-graphs")
    if psd:
        cmd.append("--psd")
    if tf32:
        cmd.append("--tf32")
    if optim:
        cmd += ["--optim", optim]
    _run(cmd)


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=7200)
def multiplayer(n_players: str = "1 2 4 8 16", modes: str = "global tile_local",
                n_diffusion_steps: int = 4, compile: bool = False, tf32: bool = False) -> None:
    """Player-count scaling: global O(p^2) vs tile_local O(p) spatial attention (random-init)."""
    cmd = ["bench_multiplayer.py", "--n-players", *n_players.split(), "--modes", *modes.split(),
           "--n-diffusion-steps", str(n_diffusion_steps)]
    if compile:
        cmd.append("--compile")
    if tf32:
        cmd.append("--tf32")
    _run(cmd)


@app.function(gpu="H100", image=engine_image, volumes={"/data": data_vol}, timeout=3600)
def engine_verify(model: str = "/data/checkpoints/mira-mini-psd/checkpoint-10000/checkpoint.pth",
                  data_index: str = "/data/datasets/rocket-science/test",
                  n_frames: int = 6, n_diffusion_steps: int = 4) -> None:
    """Phase 44: MiraEngine graph-vs-eager bit-exactness + determinism + decode sanity on a real ckpt."""
    cmd = ["python", "/root/mira-engine/examples/verify.py", "--model", model,
           "--data-index", data_index, "--n-frames", str(n_frames),
           "--n-diffusion-steps", str(n_diffusion_steps)]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/root/mira-engine", text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"engine verify failed ({proc.returncode})")


# --------------------------------------------------------------------------- training


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=3600)
def train(
    mode: str = "dit",
    batch: int = 1,
    steps: int = 30,
    compile: bool = False,
    tf32: bool = False,
    activation_ckpt: bool = False,
    size: str = "1b",
) -> None:
    """One training-throughput + MFU measurement (single H100)."""
    cmd = ["bench_train_speed.py", "--mode", mode, "--batch", str(batch), "--steps", str(steps), "--size", size]
    if compile:
        cmd.append("--compile")
    if tf32:
        cmd.append("--tf32")
    if activation_ckpt:
        cmd.append("--activation-ckpt")
    _run(cmd)


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=7200)
def ablation(mode: str = "dit", batch: int = 1, steps: int = 30) -> None:
    """The 'what did the engineering buy us' grid: baseline -> +compile -> +tf32 -> +act-ckpt."""
    rows = [
        ("baseline", []),
        ("+compile", ["--compile"]),
        ("+compile+tf32", ["--compile", "--tf32"]),
        ("+compile+tf32+actckpt", ["--compile", "--tf32", "--activation-ckpt"]),
    ]
    for name, flags in rows:
        print(f"\n===== {name} =====", flush=True)
        _run(["bench_train_speed.py", "--mode", mode, "--batch", str(batch), "--steps", str(steps)] + flags)


@app.function(gpu="H100:4", volumes={"/data": data_vol}, timeout=7200)
def ddp(mode: str = "dit", batch: int = 1, steps: int = 30, compile: bool = False) -> None:
    """DDP scaling: run the same per-GPU config on 1, 2, 4 H100s and print samples/s each."""
    base = ["bench_train_speed.py", "--mode", mode, "--batch", str(batch), "--steps", str(steps), "--per-step-barrier"]
    if compile:
        base.append("--compile")
    for n in (1, 2, 4):
        print(f"\n===== {n} GPU(s) =====", flush=True)
        _run(base, nproc=n, visible=",".join(str(i) for i in range(n)))


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=3600)
def profile(mode: str = "dit", compile: bool = False, tf32: bool = False, trace: bool = True,
            optim: str = "", streaming_cache: str = "grow") -> None:
    """torch.profiler on one step; writes a Chrome trace to the volume when trace=True."""
    cmd = ["bench_profile.py", "--mode", mode, "--streaming-cache", streaming_cache]
    if compile:
        cmd.append("--compile")
    if tf32:
        cmd.append("--tf32")
    if optim:
        cmd += ["--optim", optim]
    if trace:
        suffix = f"{mode}{'_compile' if compile else ''}{('_' + optim.replace(',', '')) if optim else ''}"
        suffix += f"_{streaming_cache}" if streaming_cache != "grow" else ""
        cmd += ["--trace-out", f"/data/trace_{suffix}.json"]
    _run(cmd)
    data_vol.commit()


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=7200)
def optim_sweep(compile: bool = True, n_frames: int = 16, variants: str = "baseline,A1,A2,A3,A4,A5") -> None:
    """Sweep the requested cast-reduction optims on inference latency, all vs the same model.

    ``variants`` is a comma list (use ``baseline`` for the no-optim run); pass e.g.
    ``--variants A3,A4,A5`` to only run the ones you don't already have. Prints one bench_infer_speed
    JSON per variant. Runs with --compile by default (the optims stack on Inductor fusion, where the
    residual ~50% cast cost lives)."""
    for v in [s.strip() for s in variants.split(",") if s.strip()]:
        optim = "" if v.lower() == "baseline" else v
        print(f"\n===== optim={v} (compile={compile}) =====", flush=True)
        cmd = ["bench_infer_speed.py", "--n-diffusion-steps", "1", "2", "4", "--n-frames", str(n_frames)]
        if compile:
            cmd.append("--compile")
        if optim:
            cmd += ["--optim", optim]
        _run(cmd)


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=10800)
def qualitycheck(checkpoint: str, optims: str = "baseline,A1,A2,A4,A5", num_samples: int = 256,
                 n_diffusion_steps: int = 4, compile: bool = False) -> None:
    """Quality gate: FDD/drift per optim variant on a REAL checkpoint (must be on the volume)."""
    cmd = ["qualitycheck_optims.py", checkpoint, "--optims", optims,
           "--num-samples", str(num_samples), "--n-diffusion-steps", str(n_diffusion_steps)]
    if compile:
        cmd.append("--compile")
    _run(cmd, extra_env={"RS_DINO_WEIGHTS_DIR": DINO_DIR})


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=10800)
def qualitycheck_steps(checkpoint: str, n_diffusion_steps: str = "1 2 4 8 10", num_samples: int = 256,
                       schedule_type: str = "linear_quadratic", compile: bool = False, data_index: str = "",
                       exclude_metric_substr: str = "auto") -> None:
    """E2 quality-vs-steps: FDD/drift at each diffusion-step count on a REAL checkpoint (on the volume).

    ``exclude_metric_substr`` keys are dropped from the MIN-VIABLE decision (metrics still run). Default
    "auto": if the DINOv3 metric weights aren't staged (stage_dino / gated), the DINO metrics are
    random-init noise, so auto-exclude 'dino fdd' and decide on latent_drift + lpips + fid_at_* (valid).
    """
    import os

    from mira.codec.dino import DINO_WEIGHT_FILENAMES

    dino_ok = os.path.exists(os.path.join(DINO_DIR, DINO_WEIGHT_FILENAMES[DINO_METRIC_MODEL]))
    if exclude_metric_substr == "auto":
        excl = [] if dino_ok else ["dino", "fdd"]
    else:
        excl = exclude_metric_substr.split()
    if not dino_ok:
        print("NOTE: DINOv3 metric weights absent -> DINO metrics are random-init; min-viable decided on "
              "latent_drift + lpips + fid_at_* (Inception, DINO-free).", flush=True)

    cmd = ["qualitycheck_steps.py", checkpoint, "--n-diffusion-steps", *n_diffusion_steps.split(),
           "--num-samples", str(num_samples), "--schedule-type", schedule_type]
    if compile:
        cmd.append("--compile")
    if data_index:
        cmd += ["--data-index", data_index]
    if excl:
        cmd += ["--exclude-metric-substr", *excl]
    # RS_DINO_WEIGHTS_DIR is still exported (harmless if empty; used when weights are staged).
    _run(cmd, extra_env={"RS_DINO_WEIGHTS_DIR": DINO_DIR})


# --------------------------------------------------------------------------- E2 asset staging


# Where staged assets live on the volume. Phase 2 (`qualitycheck_steps`) reads these paths.
CKPT_DIR = "/data/checkpoints"
DATA_DIR = "/data/datasets/rocket-science"
# The FDD *metric* builds its own DINOv3 backbone (separate from the codec's, which is restored from the
# codec checkpoint). It reads RS_DINO_WEIGHTS_DIR; with no weights there it silently uses a RANDOM-init
# DINO, making dino_frechet / dino_*_drift meaningless. Stage the real weights here for a valid gate.
DINO_DIR = "/data/dino_weights"
# Public model repos (no gate). PSD is the 2-step distilled variant the E2 plan adopts.
MODEL_REPOS = {"mira-mini": "alakazamworld/mira-mini", "mira-mini-psd": "alakazamworld/mira-mini-psd"}
# The eval dataset. Gated on the Hub (CC-BY-NC-SA) -> the HF_TOKEN secret must have accepted the terms.
DATASET_REPO = "kyutai/rocket-science"
# DINOv3 backbone the metric uses (model_size="base" -> vitb16). The .pth must be named exactly as the
# repo's DINO_WEIGHT_FILENAMES expects so torch.hub.load(weights=...) finds it. Meta-gated on the Hub.
DINO_METRIC_MODEL = "dinov3_vitb16"
DINO_HF_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _hf_snapshot(repo_id, local_dir, repo_type="model", allow_patterns=None):
    """Download an HF repo snapshot straight onto the volume (local_dir, not the cache)."""
    import os

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns,
        token=os.environ.get("HF_TOKEN"),
    )
    return path


@app.function(
    volumes={"/data": data_vol},
    timeout=10800,
    # HF_TOKEN for the gated rocket-science dataset (and higher rate limits on the public checkpoints).
    # Create it once: `modal secret create huggingface HF_TOKEN=hf_...` (a token whose account has
    # accepted the kyutai/rocket-science terms). Rename here if your secret uses a different name.
    secrets=[modal.Secret.from_name("huggingface")],
)
def stage_assets(checkpoints: bool = True, dataset: bool = True, dataset_split: str = "test") -> None:
    """E2 Phase 1: stage the mira-mini / mira-mini-psd checkpoints + rocket-science split on the volume.

    Idempotent (snapshot_download skips files already present). After it runs, use the printed paths
    with `qualitycheck_steps` (Phase 2). No GPU. `--no-checkpoints` / `--no-dataset` to stage one side.
    """
    import glob

    if checkpoints:
        for name, repo in MODEL_REPOS.items():
            dest = f"{CKPT_DIR}/{name}"
            print(f"\n===== checkpoint {repo} -> {dest} =====", flush=True)
            _hf_snapshot(repo, dest, repo_type="model")
            found = sorted(glob.glob(f"{dest}/**/checkpoint*.pth", recursive=True))
            print(f"  checkpoint files: {found or '(none found -- inspect repo layout)'}", flush=True)

    if dataset:
        print(f"\n===== dataset {DATASET_REPO} [{dataset_split}] -> {DATA_DIR} =====", flush=True)
        # Only the requested split's prefix (index.json + its shards) -- the loader reads a split dir.
        _hf_snapshot(DATASET_REPO, DATA_DIR, repo_type="dataset", allow_patterns=[f"{dataset_split}/*"])
        split_dir = f"{DATA_DIR}/{dataset_split}"
        has_index = glob.glob(f"{split_dir}/index.json")
        print(f"  data-index dir: {split_dir}  (index.json present: {bool(has_index)})", flush=True)

    data_vol.commit()
    print("\nvolume committed. Phase 2 example:\n"
          f"  modal run scripts/modal_bench.py::qualitycheck_steps \\\n"
          f"    --checkpoint {CKPT_DIR}/mira-mini-psd/<checkpoint-XXXX>/checkpoint.pth \\\n"
          f"    --n-diffusion-steps '1 2 4 8 10' --data-index {DATA_DIR}/{dataset_split}", flush=True)


@app.function(volumes={"/data": data_vol}, timeout=1800)
def verify_assets(dataset_split: str = "test") -> None:
    """Sanity-check staged assets: list checkpoint files and confirm the split index loads."""
    import glob

    print("===== staged checkpoints =====", flush=True)
    for name in MODEL_REPOS:
        found = sorted(glob.glob(f"{CKPT_DIR}/{name}/**/checkpoint*.pth", recursive=True))
        print(f"  {name}: {found or 'MISSING'}", flush=True)

    print("\n===== staged dataset =====", flush=True)
    split_dir = f"{DATA_DIR}/{dataset_split}"
    idx = f"{split_dir}/index.json"
    if glob.glob(idx):
        from pathlib import Path

        from mira.data.dataset import Index

        loaded = Index.load(Path(idx))
        print(f"  {split_dir}: index.json OK, {loaded.total_samples} samples", flush=True)
    else:
        print(f"  {split_dir}: MISSING index.json", flush=True)


@app.function(volumes={"/data": data_vol}, timeout=3600, secrets=[modal.Secret.from_name("huggingface")])
def stage_dino(hf_repo: str = DINO_HF_REPO) -> None:
    """Stage the DINOv3 metric weights onto the volume at DINO_DIR (Phase 2 prerequisite).

    Best-effort: snapshot the (Meta-gated) DINOv3 repo, then ensure a .pth exists under the exact
    filename torch.hub expects (DINO_WEIGHT_FILENAMES). If the repo only ships safetensors, this logs
    what it found so you can supply the .pth manually. Without valid weights the DINO metrics are
    random-init; Inception-FID + latent-drift stay valid (see docs).
    """
    import glob
    import os
    import shutil

    from mira.codec.dino import DINO_WEIGHT_FILENAMES

    target_name = DINO_WEIGHT_FILENAMES[DINO_METRIC_MODEL]
    os.makedirs(DINO_DIR, exist_ok=True)
    print(f"===== DINOv3 {hf_repo} -> {DINO_DIR} (target {target_name}) =====", flush=True)
    try:
        local = _hf_snapshot(hf_repo, DINO_DIR, repo_type="model")
    except Exception as exc:  # noqa: BLE001 -- gated/unauthorized/offline: don't crash the app
        print(f"  could not download ({type(exc).__name__}: {exc}).\n"
              f"  DINOv3 is Meta-gated -- request access at https://huggingface.co/{hf_repo} and make\n"
              f"  sure the HF_TOKEN secret's account is authorized, then re-run. Proceeding WITHOUT DINO\n"
              f"  metric weights is fine: qualitycheck_steps auto-excludes the DINO metrics and decides\n"
              f"  on latent_drift + lpips + fid_at_* (Inception, DINO-free).", flush=True)
        return

    target = os.path.join(DINO_DIR, target_name)
    if not os.path.exists(target):
        pths = [p for p in glob.glob(f"{local}/**/*.pth", recursive=True) if os.path.basename(p) != target_name]
        if pths:
            shutil.copy(pths[0], target)
            print(f"  copied {os.path.basename(pths[0])} -> {target_name}", flush=True)
        else:
            others = sorted(os.path.basename(p) for p in glob.glob(f"{local}/**/*", recursive=True) if os.path.isfile(p))
            print(f"  NO .pth found. Repo files: {others}\n"
                  f"  torch.hub needs a .pth state dict named {target_name}; if only safetensors are\n"
                  f"  present, convert/rename one to that path on the volume.", flush=True)
    print(f"  RS_DINO_WEIGHTS_DIR ready: {os.path.exists(target)} ({target})", flush=True)
    data_vol.commit()


@app.function(gpu="H100", volumes={"/data": data_vol}, timeout=7200)
def dataloader_test(data_index: str, batch: int = 1, num_workers: int = 6, steps: int = 30) -> None:
    """Cached-batch vs real-loader throughput (needs a clip index uploaded to the volume)."""
    _run([
        "bench_train_speed.py", "--mode", "full", "--dataloader-test",
        "--data-index", data_index, "--batch", str(batch),
        "--num-workers", str(num_workers), "--steps", str(steps),
    ])


# --------------------------------------------------------------------------- entrypoints


@app.local_entrypoint()
def main() -> None:
    """Run the core matrix: inference sweep + training ablation. (Costs H100 time.)"""
    infer.remote(compile=False)
    infer.remote(compile=True)
    ablation.remote(mode="dit")
