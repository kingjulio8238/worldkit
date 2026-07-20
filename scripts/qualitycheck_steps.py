"""E2 quality-vs-steps: how few diffusion steps can the model run before quality drops?

Latency is ~linear in `n_diffusion_steps` (with CUDA graphs: 1-step 20 ms vs 4-step 33 ms), so if
quality holds at 1-2 steps the model gets ~1.6x faster for free. The speed side is already measured
(bench_infer_speed); this measures the *quality* side: run the world-model metrics (Frechet DINO/
Inception + DINO/latent drift) at each step count on a REAL checkpoint and report the curve.

Requires a checkpoint + its dataset (gated weights) -- the random-init harness can't measure quality.
Reuses the offline-eval machinery. Usage::

    python scripts/qualitycheck_steps.py /path/to/checkpoint-XXXX/checkpoint.pth \\
        --n-diffusion-steps 1 2 4 8 --num-samples 512 --schedule-type linear
"""

from __future__ import annotations

import argparse
import json

import torch
from eval_world_model_offline import (
    _build_loader,
    load_eval_metrics_config,
    load_run_config,
    run_world_model_metrics,
)

from mira.inference.loading import load_world_model
from mira.training.checkpoints import resolve_checkpoint
from mira.world_model.config import WorldModelInferenceConfig

# FDD-family metrics where lower is better; used to pick the min viable step count.
LOWER_IS_BETTER_HINTS = ("fdd", "fid", "drift", "lpips")
# Within this fraction of the max-step (reference-quality) run counts as "quality holds".
QUALITY_TOLERANCE = 0.05  # 5%


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="checkpoint path or W&B URL")
    p.add_argument("--n-diffusion-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-samples", type=int, default=256)
    # The MIRA paper (arXiv 2607.05352, Fig 11) uses the linear-quadratic schedule + noise 0.2 for
    # few-step sampling; default to that so the sweep matches the model's few-step design.
    p.add_argument("--schedule-type", default="linear_quadratic", choices=["linear", "linear_quadratic"])
    p.add_argument("--noise-level", type=lambda s: None if s.lower() == "none" else float(s), default=0.2)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--data-index", default=None,
                   help="override the checkpoint config's dataset.test_index (e.g. the rocket-science test split)")
    p.add_argument("--codec-checkpoint", default=None,
                   help="override the config's (absolute, training-machine) codec_checkpoint. Default: "
                        "auto-detect a codec/**/checkpoint.pth shipped alongside the world-model checkpoint.")
    p.add_argument("--exclude-metric-substr", nargs="*", default=[],
                   help="drop scalar-metric keys containing any of these substrings from the MIN-VIABLE "
                        "decision (the metrics still run). Pass 'dino fdd' when the DINOv3 metric weights "
                        "are unavailable (gated) -- the DINO metrics are then random-init noise; the "
                        "decision falls back to latent_drift + lpips + fid_at_* (Inception, DINO-free).")
    return p.parse_args()


def _run_steps(model, cfg, wm_cfg, device, n_steps, schedule_type, noise_level, compile_models):
    """Run the world-model metrics for one diffusion-step count; return the scalar metric dict."""
    wm_cfg.inference = WorldModelInferenceConfig(
        n_diffusion_steps=n_steps, schedule_type=schedule_type, noise_level=noise_level
    )
    n_players = getattr(model, "n_players", 1)
    stride = wm_cfg.eval_temporal_downsampling or model.temporal_downsampling
    clip_len = model.config.n_context_frames + wm_cfg.num_unrolled_frames * stride
    loader = _build_loader(cfg, model, clip_len=clip_len, batch_size=wm_cfg.per_device_batch_size, seed=38)
    num_batches = max(1, wm_cfg.num_samples // (wm_cfg.per_device_batch_size * n_players))
    return run_world_model_metrics(
        model, iter(loader), device,
        wm_metrics_config=wm_cfg, num_eval_batches=num_batches, compile_models=compile_models,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(args.checkpoint).resolve()
    cfg = load_run_config(checkpoint)
    if args.data_index is not None:
        # The released checkpoint's config points at the training machine's dataset path; override it
        # with the local rocket-science test split (kyutai/rocket-science, gated -- hf login + accept).
        cfg.dataset.test_index = args.data_index
    wm_cfg = load_eval_metrics_config(num_samples=args.num_samples, no_compile=not args.compile)

    # The config's codec_checkpoint is an absolute path from the training machine; override it with a
    # local codec. Prefer an explicit --codec-checkpoint, else auto-detect the codec that ships in the
    # checkpoint's own directory tree (repo_root/codec/checkpoint-*/checkpoint.pth).
    codec_override = args.codec_checkpoint
    if codec_override is None:
        for anc in [checkpoint.parent, *checkpoint.parents]:
            hits = sorted(anc.glob("codec/**/checkpoint.pth"))
            if hits:
                codec_override = str(hits[0])
                break
    if codec_override is not None:
        print(f"  using codec_checkpoint: {codec_override}", flush=True)

    # One model, reused across step counts (the step count only changes the rollout sampler).
    model, _ = load_world_model(checkpoint, device=device, codec_checkpoint=codec_override)

    steps_sorted = sorted(set(args.n_diffusion_steps))
    results: dict[int, dict[str, float]] = {}
    for n_steps in steps_sorted:
        print(f"\n===== quality @ n_diffusion_steps={n_steps} =====", flush=True)
        results[n_steps] = _run_steps(
            model, cfg, wm_cfg, device, n_steps, args.schedule_type, args.noise_level, args.compile
        )
        print(json.dumps({"n_diffusion_steps": n_steps, **results[n_steps]}, indent=2), flush=True)

    # Reference = the most steps (best quality). Find the fewest steps within QUALITY_TOLERANCE.
    ref_steps = steps_sorted[-1]
    ref = results[ref_steps]
    excl = [s.lower() for s in args.exclude_metric_substr]
    candidate = [k for k in ref if any(h in k.lower() for h in LOWER_IS_BETTER_HINTS) and ref[k]]
    metric_keys = [k for k in candidate if not any(e in k.lower() for e in excl)]
    print(f"\n===== quality-vs-steps report (reference = {ref_steps} steps) =====")
    if excl:
        dropped = [k for k in candidate if any(e in k.lower() for e in excl)]
        print(f"  excluded from decision ({', '.join(excl)}): {dropped}")
    print(f"  min-viable decided on: {metric_keys}")
    min_viable = ref_steps
    for n_steps in steps_sorted:
        deltas = {k: (results[n_steps][k] - ref[k]) / abs(ref[k]) for k in metric_keys}
        holds = all(d <= QUALITY_TOLERANCE for d in deltas.values())
        tag = "OK" if holds else "DEGRADED"
        worst = max(deltas.values()) if deltas else 0.0
        print(f"  {n_steps} steps: worst metric {worst:+.1%} vs {ref_steps}-step  [{tag}]")
        if holds and n_steps < min_viable:
            min_viable = n_steps
    speedup = ref_steps / min_viable
    print(f"\n  MIN VIABLE STEPS: {min_viable} (within {QUALITY_TOLERANCE:.0%} of {ref_steps}-step quality)")
    print(f"  => ~{speedup:.2f}x latency vs {ref_steps} steps if adopted "
          f"(latency is ~linear in steps).")


if __name__ == "__main__":
    main()
