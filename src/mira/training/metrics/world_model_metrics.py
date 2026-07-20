"""World-model rollout metrics: DINO / latent drift and Frechet (DINO / Inception) curves.

:class:`WorldModelMetrics` unrolls the world model autoregressively over a held-out clip and scores
the prediction against the ground truth: per-frame PSNR / LPIPS / SSIM, DINO cosine and L2 drift,
latent cosine drift, and sliced Frechet DINO / Inception distances (an aggregate plus a per-window
curve over "frames unrolled"). The DINO and Inception backbones load lazily (``torch.hub`` /
``pytorch_fid``), so the metrics object is only constructed where those are available; the rest of
the module imports without them.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import torch
from einops import rearrange
from pydantic import BaseModel, ConfigDict

from mira.data.batch import VideoActionBatch
from mira.world_model.config import WorldModelInferenceConfig
from mira.world_model.latent_world_model import InferenceOutputs, LatentWorldModel

from .distributed_metric import DistributedMetric
from .frechet import SlicedFrechetMetric
from .image_metrics import DinoForMetrics, DistributedLPIPS, DistributedSSIM, PSNRMetric

logger = logging.getLogger(__name__)

# Standard FID uses the final 2048-dim InceptionV3 pool features.
INCEPTION_FID_DIM = 2048


def _autocast(device: torch.device | int | str):
    """bfloat16 autocast on CUDA, a no-op elsewhere (so the metrics run on CPU too)."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class WorldModelMetricsConfig(BaseModel):
    """Configuration for :class:`WorldModelMetrics` and the offline eval entry point.

    Frame counts are in LATENT-rate frames; per-frame metrics subsample the decoded video by
    ``eval_temporal_downsampling`` so the numbers are comparable across codecs with different
    ``temporal_downsampling``.
    """

    # Checkpoint / output paths, filled in by the eval entry point.
    checkpoint: str | None = None
    output_dir: str | None = None

    per_device_batch_size: int = 1  # eval batch size per GPU
    num_samples: int = 0  # limit the number of evaluated samples (0 = whole dataset)
    # Context frames the eval rollout conditions on. None keeps the model's own n_context_frames; an
    # int overrides it at eval time (see LatentWorldModel.set_inference_context), so the total eval
    # clip (n_context_frames + num_unrolled_frames * temporal_downsampling) can be shrunk to fit
    # shorter data. It must be a multiple of the codec temporal_downsampling and smaller than the
    # model window (video.timesteps).
    n_context_frames: int | None = None
    num_unrolled_frames: int = 120
    drift_metric_frames: int = 20
    fdd_slice_frames: int = 20
    # None auto-derives from model.temporal_downsampling at eval time; set explicitly to subsample
    # to a different rate (e.g. 2 for a td=1 codec running at 20 fps latent).
    eval_temporal_downsampling: int | None = None
    compile: bool = True  # whether to compile the DINO backbone for faster inference
    # Max frames per DINO forward pass, reduces peak memory (None = no limit).
    dino_max_chunk_size: int | None = None
    # Whether to build the DINOv3 backbone and compute the DINO-based metrics (dino drift + FDD). Set
    # False when DINOv3 weights are unavailable (Meta-gated): the backbone can't be constructed without
    # them, so the metric would crash. With it off, latent_drift + Inception-FID + LPIPS/PSNR/SSIM still
    # run and give a valid quality signal.
    compute_dino_metrics: bool = True
    # How many eval samples to visualize (logged to W&B from the metrics loop).
    num_viz_samples: int = 8
    # Inference settings (e.g. n_diffusion_steps, noise_level) used when unrolling the world model.
    inference: WorldModelInferenceConfig = WorldModelInferenceConfig()


class _MetricTrackers(BaseModel):
    latent_drift: DistributedMetric
    inception_frechet: SlicedFrechetMetric
    psnr: PSNRMetric
    lpips: DistributedLPIPS
    ssim: DistributedSSIM
    # DINO-based trackers are None when compute_dino_metrics is off (weights unavailable).
    dino_cos_drift: DistributedMetric | None = None
    dino_l2_drift: DistributedMetric | None = None
    dino_frechet: SlicedFrechetMetric | None = None
    # FDD decomposition: vs_recon = prediction vs codec reconstruction (world-model error);
    # codec_floor = codec reconstruction vs original frames (codec error).
    dino_frechet_vs_recon: SlicedFrechetMetric | None = None
    dino_frechet_codec_floor: SlicedFrechetMetric | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def reset(self) -> None:
        for field_name in self.__class__.model_fields:
            tracker = getattr(self, field_name)
            if tracker is not None:
                tracker.reset()


class WorldModelMetrics:
    """Stateful evaluator for world-model metrics (DINO drift, latent drift, Frechet distances).

    Usage::

        metrics = WorldModelMetrics(config, iter_dataloader, device)
        for _ in range(num_batches):
            metrics.process_batch(model)
        scalar_result, frechet_curves = metrics.compute()
    """

    def __init__(
        self,
        config: WorldModelMetricsConfig,
        iter_dataloader: Iterator[tuple[VideoActionBatch, list]],
        device: torch.device | int | str,
    ):
        # pytorch_fid is an eval-only optional dep (the `eval` extra); imported lazily so the
        # module loads without it and the offline metrics tests skip-gate on its absence.
        from pytorch_fid.inception import (  # type: ignore[import-not-found]  # noqa: PLC0415
            InceptionV3 as InceptionV3ForFID,
        )

        self.config = config
        self.iter_dataloader = iter_dataloader
        assert config.num_unrolled_frames % config.fdd_slice_frames == 0, (
            f"num_unrolled_frames ({config.num_unrolled_frames}) must be a multiple of "
            f"fdd_slice_frames ({config.fdd_slice_frames})"
        )
        self.num_slices = config.num_unrolled_frames // config.fdd_slice_frames
        # DINO backbone needs (Meta-gated) weights; skip it entirely when disabled so the eval still
        # runs on latent_drift + Inception-FID + LPIPS/PSNR/SSIM.
        self.compute_dino = config.compute_dino_metrics
        self.dino = DinoForMetrics(model_size="base").to(device) if self.compute_dino else None
        inception_block_idx = InceptionV3ForFID.BLOCK_INDEX_BY_DIM[INCEPTION_FID_DIM]
        self._inception = InceptionV3ForFID([inception_block_idx]).to(device).eval()
        self._inception.requires_grad_(False)
        tracker_kwargs = dict(
            latent_drift=DistributedMetric(device=device),
            inception_frechet=SlicedFrechetMetric(INCEPTION_FID_DIM, self.num_slices).to(device),
            psnr=PSNRMetric(device=device),
            lpips=DistributedLPIPS(device=device),
            ssim=DistributedSSIM(device=device),
        )
        if self.compute_dino:
            dd = self.dino.dino_dim
            tracker_kwargs.update(
                dino_cos_drift=DistributedMetric(device=device),
                dino_l2_drift=DistributedMetric(device=device),
                dino_frechet=SlicedFrechetMetric(dd, self.num_slices).to(device),
                dino_frechet_vs_recon=SlicedFrechetMetric(dd, self.num_slices).to(device),
                dino_frechet_codec_floor=SlicedFrechetMetric(dd, self.num_slices).to(device),
            )
        self._trackers = _MetricTrackers(**tracker_kwargs)

        if self.config.compile and self.dino is not None:
            self.dino.dino_forward = torch.compile(self.dino.dino_forward)

    def process_batch(self, model: LatentWorldModel) -> tuple[InferenceOutputs, list]:
        """Get the next batch, unroll the world model, and accumulate metrics."""
        batch, metadata = next(self.iter_dataloader)
        batch_size = batch.video.shape[0]
        device = model.device

        with torch.no_grad(), _autocast(device):
            world_model_outputs = model.inference(batch, config=self.config.inference, progress_bar=False)

            batch = batch.to(device)
            _, codec_outputs = model.codec.encode(batch.video, trim_video=False)
            # The release codec is the deterministic RAE whose posterior mean equals its sample.
            z_for_target = codec_outputs.z
            z_mean_target = rearrange(z_for_target, "b t c h w -> b t h w c")

            video = model.decode_to_video(world_model_outputs.z_t)

        # Subsample to the LATENT rate so per-frame metrics are comparable across codecs with
        # different temporal_downsampling. For td=1 stride=1 and this is a no-op.
        stride = self.config.eval_temporal_downsampling or model.temporal_downsampling
        video = video[:, ::stride]
        batch_video_eval = batch.video[:, ::stride]

        n = model.config.n_context_frames // stride
        df = self.config.drift_metric_frames
        w = self.config.fdd_slice_frames
        pred_frames = video[:, n : n + df].float()
        real_frames = batch_video_eval[:, n : n + df].float()

        self._trackers.psnr.update(pred_frames, real_frames)
        self._trackers.lpips.update(pred_frames, real_frames)
        self._trackers.ssim.update(pred_frames, real_frames)

        pred_video_full = video[:, n : n + self.num_slices * w].float()
        real_video_full = batch_video_eval[:, n : n + self.num_slices * w].float()
        for s in range(self.num_slices):
            fs = slice(s * w, (s + 1) * w)
            pred_slice = pred_video_full[:, fs]
            real_slice = real_video_full[:, fs]
            bs, ts = pred_slice.shape[:2]
            pred_flat = pred_slice.reshape(bs * ts, *pred_slice.shape[2:])
            real_flat = real_slice.reshape(bs * ts, *real_slice.shape[2:])
            with torch.no_grad():
                both_inception = self._inception(torch.cat([pred_flat, real_flat], dim=0))[0]
                both_inception = both_inception.squeeze(-1).squeeze(-1)
            pred_inception, real_inception = both_inception.chunk(2, dim=0)
            self._trackers.inception_frechet.update(s, real_inception, pred_inception)

        if self.compute_dino:
            all_dino_features = self.dino.dino_forward(
                torch.cat([video[:, n:], batch_video_eval[:, n:]], dim=0),
                max_chunk_size=self.config.dino_max_chunk_size,
            )
            pred_dino_features = all_dino_features[:batch_size]
            dino_target_features = all_dino_features[batch_size:]

            dino_similarities = torch.nn.functional.cosine_similarity(
                pred_dino_features, dino_target_features, dim=-1
            )
            # Convert cosine similarity to a distance; in practice a value > 1 means worse than chance.
            dino_cos_drifts = 1 - dino_similarities
            dino_cos_drift = dino_cos_drifts.mean(dim=(2, 3)).cpu()[:, : self.config.drift_metric_frames]

            # DINO L2 distance (MSE over feature dim, averaged over spatial/temporal).
            dino_l2_drifts = torch.nn.functional.mse_loss(
                pred_dino_features, dino_target_features, reduction="none"
            ).mean(dim=-1)
            dino_l2_drift = dino_l2_drifts.mean(dim=(2, 3)).cpu()[:, : self.config.drift_metric_frames]

            self._trackers.dino_cos_drift.update(dino_cos_drift)
            self._trackers.dino_l2_drift.update(dino_l2_drift)

        latent_similarities = torch.nn.functional.cosine_similarity(
            z_mean_target.float(),
            model.unnormalize_tokens(world_model_outputs.z_t.float()),
            dim=-1,
        )
        latent_drifts = 1 - latent_similarities
        # z_t is in LATENT frames; index by n_context_latents, not n_context_frames.
        latent_drifts = latent_drifts[:, model.n_context_latents :][:, : self.config.drift_metric_frames]
        latent_drift = latent_drifts.mean(dim=(2, 3)).cpu()
        self._trackers.latent_drift.update(latent_drift)

        if self.compute_dino:
            for s in range(self.num_slices):
                fs = slice(s * w, (s + 1) * w)
                self._trackers.dino_frechet.update(
                    s,
                    dino_target_features[:, fs].mean(dim=(-1, -2)).flatten(0, 1),
                    pred_dino_features[:, fs].mean(dim=(-1, -2)).flatten(0, 1),
                )

            # FDD decomposition: also compare against the codec's own reconstruction of the GT.
            # z_for_target is the raw codec latent, so decode it directly (rather than via
            # decode_to_video, which would also unnormalize). Same [-1, 1] -> [0, 1] post-processing.
            with torch.no_grad(), _autocast(device):
                recon_video = (model.codec.decode(z_for_target) * 0.5 + 0.5).float()
            recon_dino_features = self.dino.dino_forward(
                recon_video[:, model.config.n_context_frames :],
                max_chunk_size=self.config.dino_max_chunk_size,
            )
            for s in range(self.num_slices):
                fs = slice(s * w, (s + 1) * w)
                recon_feats = recon_dino_features[:, fs].mean(dim=(-1, -2)).flatten(0, 1)
                pred_feats = pred_dino_features[:, fs].mean(dim=(-1, -2)).flatten(0, 1)
                target_feats = dino_target_features[:, fs].mean(dim=(-1, -2)).flatten(0, 1)
                # World-model error alone: prediction vs codec reconstruction.
                self._trackers.dino_frechet_vs_recon.update(s, recon_feats, pred_feats)
                # Codec error alone: codec reconstruction vs original frames.
                self._trackers.dino_frechet_codec_floor.update(s, target_feats, recon_feats)

        return world_model_outputs, metadata

    def compute(self) -> tuple[dict[str, torch.Tensor | float], dict[str, list[float]]]:
        """Return ``(scalar metrics, per-slice Frechet curves)``, then reset.

        ``OnlineGaussian.compute()`` all_reduces, so this must run on all ranks.
        """
        df = self.config.drift_metric_frames
        scalar_result: dict[str, torch.Tensor | float] = {
            f"latent_drift_{df}": self._trackers.latent_drift.compute(),
            "psnr": self._trackers.psnr.compute_and_reset(),
            "lpips": self._trackers.lpips.compute_and_reset(),
            "ssim": self._trackers.ssim.compute_and_reset(),
        }

        # Each Frechet metric yields an aggregate scalar (pooled over all unrolled frames) plus a
        # per-slice curve.
        scalar_result["frechet_inception_distance"], fid_curve = self._trackers.inception_frechet.compute()
        frechet_curves: dict[str, list[float]] = {"fid_at": fid_curve}

        # DINO-based metrics only when the backbone was built (weights available).
        if self.compute_dino:
            scalar_result[f"dino_cos_drift_{df}"] = self._trackers.dino_cos_drift.compute()
            scalar_result[f"dino_l2_drift_{df}"] = self._trackers.dino_l2_drift.compute()
            scalar_result["frechet_dino_distance"], fdd_curve = self._trackers.dino_frechet.compute()
            scalar_result["frechet_dino_distance_vs_recon"], vr_curve = (
                self._trackers.dino_frechet_vs_recon.compute()
            )
            scalar_result["frechet_dino_distance_codec_floor"], cf_curve = (
                self._trackers.dino_frechet_codec_floor.compute()
            )
            frechet_curves["fdd_at"] = fdd_curve
            frechet_curves["fdd_at_vs_recon"] = vr_curve
            frechet_curves["fdd_at_codec_floor"] = cf_curve

        # Also expose each per-slice value as a scalar keyed by the number of frames unrolled at the
        # end of its slice (fdd_at_20 ... fdd_at_120), so the curves are trackable as W&B line plots
        # over training, not just per-step snapshots.
        w = self.config.fdd_slice_frames
        for name, values in frechet_curves.items():
            for i, v in enumerate(values):
                scalar_result[f"{name}_{(i + 1) * w}"] = v

        self._trackers.reset()
        return scalar_result, frechet_curves


def build_frechet_curve_plots(frechet_curves: dict[str, list[float]], slice_frames: int) -> dict[str, Any]:
    """Per-slice Frechet line plots (x = frames unrolled, y = Frechet distance), one per eval step."""
    import plotly.graph_objects as go  # noqa: PLC0415 -- optional dep, used only here
    import wandb  # noqa: PLC0415 -- optional dep, used only here

    metric_labels = {
        "fdd_at": "Frechet DINO Distance",
        "fid_at": "Frechet Inception Distance",
    }
    plots: dict[str, Any] = {}
    for name, values in frechet_curves.items():
        # x = number of frames unrolled at the end of each slice (20, 40, ..., 120).
        xs = [(i + 1) * slice_frames for i in range(len(values))]
        frame_ranges = [f"[{i * slice_frames}, {(i + 1) * slice_frames})" for i in range(len(values))]
        label = metric_labels.get(name, name)
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=xs,
                    y=[float(v) for v in values],
                    mode="lines+markers",
                    customdata=frame_ranges,
                    hovertemplate=(
                        "%{x} frames unrolled (slice frames %{customdata})<br>"
                        f"{label}=%{{y:.4f}}<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_layout(
            title=f"{label} vs frames unrolled",
            xaxis_title="frames unrolled",
            yaxis_title=label,
            xaxis=dict(tickmode="linear", dtick=slice_frames),
        )
        plots[f"viz/{name}"] = wandb.Plotly(fig)
    return plots
