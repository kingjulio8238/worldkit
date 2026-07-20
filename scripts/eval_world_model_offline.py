"""Run the world-model training-time evals offline from a saved checkpoint.

This reproduces, standalone, the two evaluations that ``train_world_model.py`` runs periodically:

  1. Validation loss   (mirrors ``run_validation``): diffusion loss on the test set.
  2. World-model metrics (mirrors the metrics loop): DINO / latent drift + Frechet DINO/Inception
     curves + per-frame PSNR/LPIPS/SSIM, with optional rollout viz clips saved to disk.

The dataset / model config is read from the ``world_model_config.yaml`` saved alongside the
checkpoint, so the eval matches the training run; how the metrics eval is run comes from
``configs/eval_world_model.yaml``. Single-process (no torchrun); video-only (no audio metrics).

The checkpoint may be a local path or a W&B run -- anything ``resolve_checkpoint`` accepts. Usage::

    python scripts/eval_world_model_offline.py /path/to/checkpoint-1000/checkpoint.pth
    python scripts/eval_world_model_offline.py <wandb-url> --num-samples 512 --skip-validation
    python scripts/eval_world_model_offline.py <ckpt> --viz 8 --output-dir out --no-compile
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from mira.data.batch import VideoActionBatch
from mira.inference.rollout import measure_rollout_speed
from mira.training.metrics.distributed_metric import DistributedMetric
from mira.training.metrics.world_model_metrics import WorldModelMetricsConfig
from mira.world_model.config import WorldModelInferenceConfig

if TYPE_CHECKING:
    # Single- and multi-player models are duck-typed under LatentWorldModel (the wrapper exposes the
    # same forward/inference/visualize/decode_to_video surface); n_players is read with getattr.
    from mira.world_model.latent_world_model import LatentWorldModel

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "world_model_config.yaml"
EVAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "eval_world_model.yaml"
SPEED_BENCH_FRAMES = 32


def _autocast(device: torch.device | int | str):
    """bfloat16 autocast on CUDA, a no-op elsewhere (so the eval pieces run on CPU too)."""
    from contextlib import nullcontext

    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_run_config(checkpoint_path: Path):
    """Load the training run's config saved alongside the checkpoint (two dirs above the .pth)."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    config_path = checkpoint_path.parents[1] / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find {CONFIG_FILENAME} at {config_path}. It is expected two directories "
            "above the checkpoint (the training output dir)."
        )
    return OmegaConf.load(config_path)


def load_eval_metrics_config(
    *,
    num_samples: int | None = None,
    no_compile: bool = False,
    inference: WorldModelInferenceConfig | None = None,
    compute_dino_metrics: bool | None = None,
) -> WorldModelMetricsConfig:
    """Build the eval :class:`WorldModelMetricsConfig` from ``configs/eval_world_model.yaml``.

    CLI overrides (sample count, compile, inference knobs, DINO on/off) are applied on top.
    """
    from hydra.utils import instantiate  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    raw = OmegaConf.load(EVAL_CONFIG)
    config: WorldModelMetricsConfig = instantiate(raw.world_model_metrics)
    if num_samples is not None:
        config.num_samples = num_samples
    if no_compile:
        config.compile = False
    if inference is not None:
        config.inference = inference
    if compute_dino_metrics is not None:
        config.compute_dino_metrics = compute_dino_metrics
    return config


def run_validation_loss(
    model: "LatentWorldModel",
    val_iter: Iterator[tuple[VideoActionBatch, Any]],
    device: torch.device | int | str,
    n_batches: int,
) -> dict[str, float]:
    """Mirror ``train_world_model.run_validation``: average the forward diffusion loss on the test set."""
    import tqdm  # noqa: PLC0415

    model.eval()
    metric_trackers: dict[str, DistributedMetric] = defaultdict(lambda: DistributedMetric(device=device))
    t1 = time.time()
    for _ in tqdm.trange(max(1, n_batches), desc="Validation loss"):
        batch, _ = next(val_iter)
        batch = batch.to(device)
        with torch.no_grad(), _autocast(device):
            for k, v in model(batch).items():
                metric_trackers[k].update(v)

    metrics = {k: tracker.compute_and_reset().item() for k, tracker in metric_trackers.items()}
    logger.info("Validation loss took %.1fs over %d batches", time.time() - t1, max(1, n_batches))
    return metrics


def save_viz_videos(
    model: "LatentWorldModel",
    inference_outputs,
    metadata: list,
    output_dir: Path,
    sample_idx: int,
) -> None:
    """Render one rollout to disk as a predicted-over-ground-truth grid (video-only, no audio)."""
    from mira.training.visualization import (  # noqa: PLC0415
        draw_text_on_first_frame,
        videos_to_grid,
        write_video_ffmpeg,
    )

    n_players = getattr(model, "n_players", 1)
    captions = [f"{m.match_id}:{m.perspective}" for m in metadata]
    grouped = [" + ".join(captions[i : i + n_players]) for i in range(0, len(captions), n_players)]

    viz_video = model.visualize(inference_outputs)["viz_video"]
    viz_video = draw_text_on_first_frame(viz_video, grouped)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_video_ffmpeg(
        output_dir / f"viz_{sample_idx:03d}.mp4",
        videos_to_grid(viz_video.cpu()),
        fps=model.config.video.fps,
    )


def run_world_model_metrics(
    model: "LatentWorldModel",
    metrics_iter: Iterator[tuple[VideoActionBatch, list]],
    device: torch.device | int | str,
    *,
    wm_metrics_config: WorldModelMetricsConfig,
    num_eval_batches: int,
    num_viz: int = 0,
    output_dir: Path | None = None,
    compile_models: bool = False,
) -> dict[str, float]:
    """Mirror the training metrics loop: unroll, score, optionally save viz, then aggregate."""
    import tqdm  # noqa: PLC0415

    from mira.training.metrics.world_model_metrics import WorldModelMetrics  # noqa: PLC0415

    if compile_models and not getattr(model, "_offline_eval_compiled", False):
        model.world_model.compile()
        model.decode_to_video = torch.compile(model.decode_to_video)
        # Guard so the sweep doesn't re-wrap decode on every call; setattr sidesteps nn.Module's
        # typed __setattr__ (which only accepts Tensor/Module values).
        setattr(model, "_offline_eval_compiled", True)

    wm_metrics = WorldModelMetrics(config=wm_metrics_config, iter_dataloader=metrics_iter, device=device)
    model.eval()
    t_start = time.time()
    for batch_idx in tqdm.trange(num_eval_batches, desc="World model metrics"):
        inference_outputs, metadata = wm_metrics.process_batch(model)
        if batch_idx < num_viz and output_dir is not None:
            save_viz_videos(model, inference_outputs, metadata, output_dir, batch_idx)

    scalar_result, _frechet_curves = wm_metrics.compute()
    elapsed = time.time() - t_start
    logger.info("World model metrics took %.1fs over %d batches", elapsed, num_eval_batches)
    if num_viz > 0 and output_dir is not None:
        logger.info("Saved %d viz clip(s) to %s", min(num_viz, num_eval_batches), output_dir)
    return {k: float(v) for k, v in scalar_result.items()}


def measure_denoise_speed(
    model: "LatentWorldModel",
    batch: VideoActionBatch,
    config: WorldModelInferenceConfig | None = None,
    n_frames: int = SPEED_BENCH_FRAMES,
) -> dict[str, float]:
    """Time the pure denoising rollout (no decode / metrics) at batch size 1, after a short warmup."""
    config = config or WorldModelInferenceConfig()
    with torch.no_grad(), _autocast(model.device):
        measure_rollout_speed(model, batch, config, n_frames=min(4, n_frames))  # warmup
        return measure_rollout_speed(model, batch, config, n_frames=n_frames)


def _frame_size(cfg) -> tuple[int, int] | None:
    fs = cfg.dataset.get("frame_size")
    return tuple(fs) if fs is not None else None  # type: ignore[return-value]


def _build_loader(cfg, model: "LatentWorldModel", *, clip_len: int, batch_size: int, seed: int):
    """Build a held-out eval loader from the checkpoint's dataset config (fixed seed, no replays)."""
    from mira.data.training_loader import create_loader  # noqa: PLC0415

    n_players = getattr(model, "n_players", 1)
    return create_loader(
        index_path=cfg.dataset.test_index,
        clip_len=clip_len,
        target_fps=model.config.video.fps,
        action_fps=model.config.actions.target_fps,
        n_players=n_players,
        batch_size=batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle_buffer_size=cfg.dataloader.shuffle_buffer_size,
        frame_size=_frame_size(cfg),
        valid_keys=list(model.config.actions.valid_keys),
        seed=seed,
        exclude_replays=True,
        infinite=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=str, help="Checkpoint path or W&B run URL.")
    parser.add_argument("--num-samples", type=int, default=None, help="World-model-metrics samples.")
    parser.add_argument("--val-n-samples", type=int, default=None, help="Validation-loss samples.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the validation-loss eval.")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip the world-model-metrics eval.")
    parser.add_argument("--viz", type=int, default=0, metavar="N", help="Render N rollout viz clips.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write viz clips (default: <checkpoint dir>/offline_eval).",
    )
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument(
        "--n-diffusion-steps", type=int, default=None, help="Override rollout diffusion steps."
    )
    parser.add_argument("--schedule-type", type=str, default=None, choices=["linear", "linear_quadratic"])
    parser.add_argument(
        "--noise-level",
        type=lambda s: None if s.lower() == "none" else float(s),
        default=argparse.SUPPRESS,
        help="kv-cache noise level; 'none' merges the cache update into the last diffusion step.",
    )
    return parser.parse_args()


def _inference_overrides(
    args_dict: dict[str, Any], base: WorldModelInferenceConfig
) -> WorldModelInferenceConfig:
    """Apply the CLI rollout knobs onto the eval config's inference defaults (only those provided)."""
    updates: dict[str, Any] = {}
    if args_dict.get("n_diffusion_steps") is not None:
        updates["n_diffusion_steps"] = args_dict["n_diffusion_steps"]
    if args_dict.get("schedule_type") is not None:
        updates["schedule_type"] = args_dict["schedule_type"]
    if "noise_level" in args_dict:  # argparse.SUPPRESS: present only when the flag was passed
        updates["noise_level"] = args_dict["noise_level"]
    return base.model_copy(update=updates) if updates else base


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    args_dict = vars(args)  # so `in args` works for the SUPPRESS-defaulted noise_level

    from mira.inference.loading import load_world_model  # noqa: PLC0415
    from mira.training.checkpoints import resolve_checkpoint  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(args.checkpoint).resolve()
    cfg = load_run_config(checkpoint)
    output_dir = args.output_dir or (checkpoint.parent / "offline_eval")

    model, _ = load_world_model(checkpoint, device=device)
    model = model.eval()
    logger.info("Loaded world model from %s on %s", checkpoint, device)

    eval_config = load_eval_metrics_config(
        num_samples=args.num_samples,
        no_compile=args.no_compile,
    )
    eval_config.inference = _inference_overrides(args_dict, eval_config.inference)
    compile_models = (not args.no_compile) and bool(cfg.run.get("compile"))

    results: dict[str, float] = {}

    if not args.skip_validation:
        batch_size = (cfg.validation.batch_size or cfg.run.batch_size) * getattr(model, "n_players", 1)
        n_samples = args.val_n_samples if args.val_n_samples is not None else cfg.validation.val_n_samples
        val_loader = _build_loader(
            cfg, model, clip_len=model.config.video.timesteps * 2, batch_size=batch_size, seed=37
        )
        results |= {
            f"test/{k}": v
            for k, v in run_validation_loss(
                model, iter(val_loader), device, n_batches=max(1, n_samples // batch_size)
            ).items()
        }

    if not args.skip_metrics:
        # Apply the eval's context override before deriving the clip length, so the loader, the
        # rollout, and the metric indexing all agree on n_context_frames (see set_inference_context).
        if eval_config.n_context_frames is not None:
            model.set_inference_context(eval_config.n_context_frames)
        stride = eval_config.eval_temporal_downsampling or model.temporal_downsampling
        metrics_loader = _build_loader(
            cfg,
            model,
            clip_len=model.config.n_context_frames + eval_config.num_unrolled_frames * stride,
            batch_size=eval_config.per_device_batch_size,
            seed=38,
        )
        num_eval_batches = max(1, eval_config.num_samples // eval_config.per_device_batch_size)
        results |= {
            f"metrics/{k}": v
            for k, v in run_world_model_metrics(
                model,
                iter(metrics_loader),
                device,
                wm_metrics_config=eval_config,
                num_eval_batches=num_eval_batches,
                num_viz=args.viz,
                output_dir=output_dir,
                compile_models=compile_models,
            ).items()
        }
        # Pure denoising speed (bs=1, no decode/metrics).
        try:
            speed_batch = _build_loader(
                cfg,
                model,
                clip_len=model.config.n_context_frames + SPEED_BENCH_FRAMES * model.temporal_downsampling,
                batch_size=1,
                seed=39,
            )
            batch, _ = next(iter(speed_batch))
            batch = batch.to(device)
            model.codec.preprocess_batch(batch)
            results |= {
                f"metrics/{k}": v
                for k, v in measure_denoise_speed(model, batch, eval_config.inference).items()
            }
        except Exception:
            logger.exception("Denoise-speed measurement failed; skipping it.")

    logger.info("%s", "=" * 60)
    logger.info("Offline eval results:")
    for k, v in results.items():
        logger.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()
