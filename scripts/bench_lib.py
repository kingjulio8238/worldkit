"""Shared helpers for the training / inference benchmarks.

The benchmarks run on a **random-initialised** model built from the repo's own config files, so no
(potentially gated) checkpoint is needed: speed, MFU and latency are weight-agnostic. The only thing
random weights change is output *quality*, which the speed harness does not measure.

The tricky bit is that :class:`~mira.world_model.latent_world_model.LatentWorldModel` requires a
frozen codec checkpoint at construction time. :func:`build_random_world_model` sidesteps that by
building a random :class:`~mira.codec.codec_model.VideoCodec` (DINOv3 backbone via ``torch.hub`` with
``pretrained=False`` -- code only, no gated weights) and injecting it through the codec-load hook, so
the real ``LatentWorldModel.__init__`` runs unchanged.

Everything here is plain importable Python (no Modal, no CUDA required to import) so the same code
runs locally for linting and on a Modal H100 for the real numbers.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Callable

import torch
from omegaconf import OmegaConf

from mira.codec.codec_model import VideoCodec
from mira.codec.config import VideoCodecConfig
from mira.data.batch import VideoActionBatch
from mira.ml.config_loading import strip_hydra_targets
from mira.world_model.actions_config import ActionConfig, ActionTensors

# Repo layout: this file is <repo>/scripts/bench_lib.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

# H100 SXM peak bf16 (no sparsity), FLOP/s. Override per hardware.
H100_SXM_BF16_PEAK_FLOPS = 989.5e12


def unwrap(module):
    """Return the underlying module, unwrapping a DistributedDataParallel wrapper if present."""
    return getattr(module, "module", module)


# --------------------------------------------------------------------------- model construction


def _load_codec_config(codec_config_name: str = "raev2_codec_tdown") -> VideoCodecConfig:
    """Build a ``VideoCodecConfig`` from a repo codec config yaml, resolving its ``${run.*}`` refs."""
    raw = OmegaConf.load(CONFIGS_DIR / "model" / f"{codec_config_name}.yaml")
    # The codec config references ${run.compile}; supply a run node so interpolations resolve.
    merged = OmegaConf.merge({"run": {"compile": False}}, raw)
    container = OmegaConf.to_container(merged.architecture.config, resolve=True)
    assert isinstance(container, dict)
    return VideoCodecConfig.model_validate(strip_hydra_targets(container))


def _compose_world_model_config(size: str, multiplayer: bool):
    """Return the resolved ``model.architecture.config`` node for the requested size.

    Uses Hydra compose against the repo configs so the size overlay (e.g. ``1b``) and the
    dataset/action defaults are applied exactly as training does.
    """
    from hydra import compose, initialize_config_dir  # noqa: PLC0415 -- optional dep

    config_name = "train_world_model"
    overrides: list[str] = []
    if size != "1b":
        # The size overlay is packaged at model/latent_world_model@architecture.config.
        overrides.append(f"model/latent_world_model@model.architecture.config={size}")
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg.model.architecture.config


def build_random_world_model(
    size: str = "1b",
    device: str | torch.device = "cuda",
    *,
    codec_config_name: str = "raev2_codec_tdown",
    latent_mean_std: tuple[float, float] = (0.0, 1.0),
    wm_config_overrides: dict | None = None,
):
    """Construct a random-init :class:`LatentWorldModel` with a random-init frozen codec.

    No checkpoint is read. The codec's DINOv3 backbone is built with ``pretrained=False`` (needs
    ``torch.hub`` network/cache for the *code*, not gated weights). ``latent_mean_std`` is injected
    into the world-model config so normalization is well-defined without a checkpoint.
    """
    from mira.world_model import latent_world_model as lwm_module  # noqa: PLC0415
    from mira.world_model.config import LatentWorldModelConfig  # noqa: PLC0415

    wm_cfg_node = _compose_world_model_config(size, multiplayer=False)
    wm_cfg_dict = lwm_module._config_dict_from_yaml(wm_cfg_node)
    # A non-null placeholder so __init__ takes the codec-load path (which we hook below); its value
    # is never dereferenced because resolve_checkpoint is patched to a passthrough.
    wm_cfg_dict["codec_checkpoint"] = "__random__"
    wm_cfg_dict["latent_mean_std"] = list(latent_mean_std)
    if wm_config_overrides:
        wm_cfg_dict.update(wm_config_overrides)  # e.g. the A1-A3 optimization flags
    wm_config = LatentWorldModelConfig.model_validate(wm_cfg_dict)

    codec_cfg = _load_codec_config(codec_config_name)
    random_codec = VideoCodec(codec_cfg, require_dino_weights=False)
    random_codec.info_from_checkpoint = {"latent_mean_std": list(latent_mean_std)}

    # Hook the two functions __init__ uses to obtain the frozen codec so the *real* constructor runs
    # unchanged against our random codec. Restored in `finally` so we never leak the patch. The
    # original classmethod descriptor is read from __dict__ so it is restored as a classmethod, not
    # a bound method.
    orig_resolve = lwm_module.resolve_checkpoint
    orig_load = VideoCodec.__dict__["load_from_checkpoint"]
    lwm_module.resolve_checkpoint = lambda ckpt: ckpt  # type: ignore[assignment]
    VideoCodec.load_from_checkpoint = classmethod(  # type: ignore[assignment]
        lambda cls, *a, **k: random_codec
    )
    try:
        model = lwm_module.LatentWorldModel(wm_config)
    finally:
        lwm_module.resolve_checkpoint = orig_resolve  # type: ignore[assignment]
        VideoCodec.load_from_checkpoint = orig_load  # type: ignore[assignment]

    model.to(device)
    return model


# --------------------------------------------------------------------------- synthetic data


def make_synthetic_batch(
    model,
    batch_size: int,
    device: str | torch.device = "cuda",
    n_video_frames: int | None = None,
) -> VideoActionBatch:
    """A random ``VideoActionBatch`` sized to the model's training window (full video+action stream).

    Video is uint8 ``(B, T, C, H, W)`` at the model's native resolution; actions are random multi-hot
    key presses over the full window. This is what ``model(batch)`` (the end-to-end forward, incl.
    the codec DINO encode) consumes. Pass ``n_video_frames`` to size a longer clip (e.g. for a
    rollout that unrolls more frames than the training window); it must be even for the td=2 codec.
    """
    cfg = model.config
    t = n_video_frames if n_video_frames is not None else cfg.video.timesteps
    video = torch.randint(
        0, 256, (batch_size, t, cfg.video.channels, cfg.video.height, cfg.video.width),
        dtype=torch.uint8, device=device,
    )
    # Actions are frame-indexed; the loader provides target_fps//video_fps actions per frame. The
    # world model reads slice_time windows up to n_action_steps+off, so provide the full stream.
    n_action_frames = t * cfg.actions.target_fps // cfg.video.fps
    n_keys = len(cfg.actions.valid_keys)
    actions = ActionTensors(
        config=ActionConfig(valid_keys=list(cfg.actions.valid_keys), target_fps=cfg.actions.target_fps),
        batch_size=batch_size,
    )
    actions.key_presses = torch.randint(
        0, 2, (batch_size, n_action_frames, n_keys), dtype=torch.int32, device=device
    )
    actions.mouse_movements = torch.zeros((batch_size, n_action_frames, 2), dtype=torch.float32, device=device)
    actions.game_mouse_sensitivity = torch.full((batch_size,), float("nan"), device=device)
    return VideoActionBatch(video=video, actions=actions)


def make_synthetic_latents(model, batch_size: int, device: str | torch.device = "cuda"):
    """Random latents ``z`` (b, t, h, w, c) and action embedding ``a`` for the DiT-only training tail.

    These feed ``model.diffusion_loss(z, a)`` directly, isolating the trainable diffusion transformer
    from the frozen codec's DINO encode.
    """
    dit = unwrap(model.world_model)
    t = dit.n_latent_frames
    z = torch.randn(batch_size, t, dit.latent_height, dit.latent_width, model.latent_dim, device=device)
    # Action embedding as produced by action_encoder: (b, t_actions, hidden_dim). The diffusion loss
    # expects one action token per latent frame.
    a = torch.randn(batch_size, t, model.config.hidden_dim, device=device)
    return z, a


# --------------------------------------------------------------------------- FLOP / MFU accounting


def count_trainable_params(model) -> int:
    """Parameters that receive gradients (the DiT + action encoder + bos); excludes the frozen codec."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _gqa_sdpa_flop_count(query_shape, key_shape, value_shape):
    """GQA-tolerant SDPA FLOP count: use the query head count (KV heads broadcast up under GQA).

    torch <= 2.8's built-in ``sdpa_flop_count`` asserts ``h_q == h_k``, which trips on grouped-query
    attention (``enable_gqa=True``, num_kv_heads != num_heads). This replacement drops that assert and
    counts against the query heads, which is what the attention actually computes.
    """
    b, h_q, s_q, d_q = query_shape
    _b2, _h_k, s_k, _d_k = key_shape
    _b3, _h_v, _s_v, d_v = value_shape
    return 2 * b * h_q * s_q * s_k * d_q + 2 * b * h_q * s_q * s_k * d_v  # QK^T + attn@V (×2 MACs)


def measure_forward_flops(forward_fn: Callable[[], object]) -> int:
    """Measured forward FLOPs of ``forward_fn`` via ``torch.utils.flop_counter``.

    Counts matmul / conv / scaled_dot_product_attention flops, so it handles the factored
    spatial/temporal attention correctly (a hand-derived 6ND would miss the factoring). Run once,
    outside the timed loop. Patches the SDPA formula to be GQA-tolerant for the run (see
    :func:`_gqa_sdpa_flop_count`) and restores it after.
    """
    from torch.utils import flop_counter as _fc  # noqa: PLC0415
    from torch.utils.flop_counter import FlopCounterMode  # noqa: PLC0415

    orig = getattr(_fc, "sdpa_flop_count", None)
    if orig is not None:
        _fc.sdpa_flop_count = _gqa_sdpa_flop_count
    try:
        with FlopCounterMode(display=False) as fcm:
            forward_fn()
        return fcm.get_total_flops()
    finally:
        if orig is not None:
            _fc.sdpa_flop_count = orig


def analytic_forward_flops(n_params: int, tokens_per_forward: int) -> int:
    """Fallback forward-FLOP estimate: ~2·N·tokens (matmul-dominated; attention is small here).

    Used only if the measured path fails, so the bench never crashes on FLOP counting. Validated
    against the measured value on this model (measured 6.75 TFLOP vs 2·N·tokens = 6.85 TFLOP, ~1.5%).
    """
    return 2 * n_params * tokens_per_forward


def training_flops_per_step(forward_flops: int) -> int:
    """Fwd+bwd training FLOPs from measured forward FLOPs (backward ≈ 2x forward, the standard 3x)."""
    return 3 * forward_flops


def mfu(flops_per_step: float, steps_per_sec: float, peak_flops: float = H100_SXM_BF16_PEAK_FLOPS) -> float:
    """Model FLOPs Utilization in [0, 1]."""
    return (flops_per_step * steps_per_sec) / peak_flops


# --------------------------------------------------------------------------- timing


@contextlib.contextmanager
def cuda_timer():
    """Context manager yielding a one-element list that receives the elapsed wall-seconds (CUDA-synced)."""
    out: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)


def time_loop(step_fn: Callable[[], object], n_iters: int, warmup: int = 3) -> dict[str, float]:
    """Time ``step_fn`` over ``n_iters`` iterations after ``warmup`` untimed iterations.

    Returns ``{"ms_per_iter", "iters_per_sec"}``. The caller multiplies by batch size for samples/s.
    """
    for _ in range(warmup):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"ms_per_iter": elapsed / n_iters * 1000, "iters_per_sec": n_iters / elapsed}


def peak_mem_gb() -> float:
    """Peak CUDA memory allocated since the last reset, in GB (0 on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def enable_tf32(enabled: bool) -> None:
    """Toggle TF32 for the fp32 matmuls (QK-norm, RoPE). No-op safe on CPU."""
    if enabled:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def autocast_ctx(device: str | torch.device = "cuda"):
    """bf16 autocast on CUDA, no-op elsewhere (mirrors the trainer)."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
