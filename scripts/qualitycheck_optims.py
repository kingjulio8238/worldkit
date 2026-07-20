"""Quality gate for the A1-A5 inference optimizations: does any of them regress world-model quality?

Loads a REAL checkpoint (speed is weight-agnostic but *quality* is not), runs the same world-model
metrics the trainer/offline-eval use (Frechet DINO/Inception + latent/DINO drift) with each optim
variant toggled on, and reports the metrics side-by-side so a precision-changing optim (A2 bf16 RoPE,
A5 bf16 QK-norm) can be accepted or rejected. A1 and A3 are equivalence-preserving and should match
baseline within noise; A5 is expected to move the numbers.

Requires a checkpoint + its dataset (the released weights are gated) -- this is the only step of the
"secure A" plan that can't run on random init. Usage::

    python scripts/qualitycheck_optims.py /path/to/checkpoint-XXXX/checkpoint.pth \\
        --optims baseline,A1,A2,A4,A5 --num-samples 512 --n-diffusion-steps 4

Reloads the model fresh per variant so no optim's in-place state leaks into the next.
"""

from __future__ import annotations

import argparse
import json

import torch
from bench_optims import apply_optims, parse_optim_spec
from eval_world_model_offline import (
    _build_loader,
    load_eval_metrics_config,
    load_run_config,
    run_world_model_metrics,
)

from mira.inference.loading import load_world_model
from mira.training.checkpoints import resolve_checkpoint
from mira.world_model.config import WorldModelInferenceConfig

# FDD-family metrics where lower is better; used to flag regressions vs baseline.
LOWER_IS_BETTER_HINTS = ("fdd", "fid", "drift", "lpips")
REGRESSION_REL_THRESHOLD = 0.02  # >2% worse than baseline on a lower-is-better metric = flag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="checkpoint path or W&B URL")
    p.add_argument("--optims", default="baseline,A1,A2,A4,A5", help="comma list of variants to compare")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--n-diffusion-steps", type=int, default=4)
    p.add_argument("--schedule-type", default="linear", choices=["linear", "linear_quadratic"])
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def _run_variant(checkpoint, cfg, wm_cfg, device, variant: str, compile_models: bool) -> dict[str, float]:
    """Load a fresh model, toggle the variant's optims, run the metrics, return the scalar dict."""
    model, _ = load_world_model(checkpoint, device=device)
    if variant != "baseline":
        apply_optims(model, parse_optim_spec(variant))
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
    wm_cfg = load_eval_metrics_config(
        num_samples=args.num_samples,
        no_compile=not args.compile,
        inference=WorldModelInferenceConfig(
            n_diffusion_steps=args.n_diffusion_steps, schedule_type=args.schedule_type
        ),
    )

    variants = [v.strip() for v in args.optims.split(",") if v.strip()]
    results: dict[str, dict[str, float]] = {}
    for variant in variants:
        print(f"\n===== quality: {variant} =====", flush=True)
        results[variant] = _run_variant(checkpoint, cfg, wm_cfg, device, variant, args.compile)
        print(json.dumps(results[variant], indent=2), flush=True)

    # Regression report vs baseline on the lower-is-better metrics.
    base = results.get("baseline")
    print("\n===== regression report (vs baseline) =====")
    if base is None:
        print("no baseline variant run; skipping comparison")
        return
    for variant in variants:
        if variant == "baseline":
            continue
        flags = []
        for k, v in results[variant].items():
            if base.get(k, 0) and any(h in k.lower() for h in LOWER_IS_BETTER_HINTS):
                rel = (v - base[k]) / abs(base[k])
                mark = "  ⚠️ REGRESSION" if rel > REGRESSION_REL_THRESHOLD else ""
                flags.append(f"    {k}: {base[k]:.4f} -> {v:.4f} ({rel:+.1%}){mark}")
        verdict = "REGRESSION" if any("REGRESSION" in f for f in flags) else "ok"
        print(f"  {variant}: {verdict}")
        for f in flags:
            print(f)


if __name__ == "__main__":
    main()
