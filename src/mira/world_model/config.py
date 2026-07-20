"""Configuration schemas for the latent world model and its inference.

:class:`LatentWorldModelConfig` is the architecture/training config (validated from the checkpoint's
saved yaml); :class:`WorldModelInferenceConfig` holds the autoregressive-rollout sampling knobs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from mira.ml.image_config import ImageConfig
from mira.world_model.actions_config import ActionConfig


class LatentWorldModelConfig(BaseModel):
    """Architecture and training configuration of the :class:`LatentWorldModel`.

    The world model is a flow-matching diffusion transformer operating on the frozen codec's latent
    grid. The codec is loaded from ``codec_checkpoint`` and frozen; its latent dimension, temporal
    and spatial downsampling factors are read from it rather than configured here.
    """

    model_config = ConfigDict(extra="forbid")

    actions: ActionConfig
    video: ImageConfig
    codec_checkpoint: str | None = None
    # Optional override of the codec latent mean/std; normally read from the codec checkpoint.
    latent_mean_std: list[float] | None = None

    causal: bool = True
    # Activation checkpointing to bound the memory peak of grad-carrying forward passes.
    # False = off; True = checkpoint every grad-carrying pass; "psd-only" = checkpoint only the PSD
    # student pass (the teacher passes are no_grad and store nothing), leaving the diagonal diffusion
    # pass un-checkpointed at full speed while still bounding the activation peak of PSD steps.
    activation_checkpointing: bool | Literal["psd-only"] = False
    # PSD-M self-distillation: progressive self-distillation with midpoint targets; see "How to
    # build a consistency model" (https://arxiv.org/abs/2505.18825) and the equivalent scheme in
    # "One Step Diffusion via Shortcut Models" (https://arxiv.org/abs/2410.12557).
    # Two mutually-exclusive ways to mix in the PSD loss (both 0 = PSD disabled):
    #   - psd_loss_prob: STOCHASTIC. On a fraction `psd_loss_prob` of steps, add the (unweighted)
    #     PSD loss to the total; the other steps skip its three extra forward passes. Cheaper on
    #     average; the gradient is a stochastic estimate.
    #   - psd_weight: DETERMINISTIC. Compute the PSD loss every step and add `psd_weight * loss_psd`
    #     to the total. Exact gradient, but always pays the three extra forward passes.
    psd_loss_prob: float = 0.0
    psd_weight: float = 0.0
    # If True, condition each denoising step on the clean latents of the previous frames
    # (adds the past_proj and bos parameters). Avoids the train/inference mismatch in diffusion
    # forcing. The default keeps these parameters out of the state dict for strict loading.
    use_clean_past: bool = False

    dropout_action_prob: float = 0.1
    # Per-player subset-key action dropout (multiplayer): when on, a dropped (player-)row drops
    # either all keys or just the canonical subset (Q/E/Space/Shift/Ctrl). Default off -> legacy
    # whole-keyboard dropout, so single-player models are unchanged.
    dropout_action_per_player: bool = False
    action_subset_drop_prob: float = 0.5
    learned_temporal_pool: bool = False
    # If True, use the posterior mean from the codec encoder instead of a sampled latent. For the
    # deterministic RAE codec the posterior mean equals the sample, so this reduces to reading z.
    use_codec_posterior_mean: bool = False

    n_register_tokens: int = 0

    # Multiplayer scaling. The multiplayer wrapper tiles n_players clips into one grid (stacked along
    # height); "global" spatial attention over all p*h*w tokens is O(p^2). "tile_local" runs
    # block-diagonal spatial attention within each player's tile + a cheap cross-player mixer over
    # pooled per-tile summaries -> O(p). n_spatial_tiles is how many player tiles the height splits into
    # (set by MultiWrapperWorldModel = n_players; 1 = single grid, tile_local is a no-op). Default
    # "global" preserves the shipped single-player and multiplayer behaviour exactly.
    spatial_attention: Literal["global", "tile_local"] = "global"
    n_spatial_tiles: int = 1

    patch_size: int = 1
    attention_gating: bool = False
    # If True, apply AdaLN conditioning to the attention sublayers (space/time) as well as the MLP,
    # as in standard DiT. The default applies MLP-only conditioning and keeps the extra parameters
    # out of the state dict for strict loading.
    ada_attn_ln: bool = False

    n_context_frames: int = 39  # only used during inference

    hidden_dim: int = 2048
    n_head: int = 16
    n_kv_head: int | None = None
    n_layers: int = 16
    time_attention_every: int = 1

    # --- Inference RoPE/QK-norm optimizations. Only A3 (rope_precompute) is on by default: it is a
    # bit-exact −22% per-frame denoise win on H100 (see docs/optimization_plan.md). A1/A2 are off:
    # the de-noised benchmark showed A1 is within noise and A2 (bf16 RoPE, a precision change) is
    # redundant once RoPE tables are precomputed. ---
    # A1: compute QK-norm via the fused F.layer_norm/F.rms_norm kernel (in-kernel fp32 accum, no
    # materialized fp32 tensor) instead of the explicit fp32 round-trip. No measured benefit under
    # compile; left off.
    qk_norm_fused: bool = False
    # A2: apply RoPE without upcasting q/k to fp32 (rotate in the input dtype). Set False for bf16 RoPE.
    # Off by default (a precision change that adds nothing on top of A3).
    rope_upcast: bool = True
    # A3: precompute the RoPE cos/sin tables into buffers and slice per call. Bit-exact and
    # compile-friendly (the recompute path triggers torch.compile recompiles from a runtime cache).
    # On by default -- the measured, precision-preserving inference win.
    rope_precompute: bool = True

    @property
    def psd_enabled(self) -> bool:
        """Whether the PSD-M loss is active (via either the stochastic or weighted path)."""
        return self.psd_loss_prob > 0 or self.psd_weight > 0

    @model_validator(mode="after")
    def _check_psd_exclusive(self) -> LatentWorldModelConfig:
        if self.psd_loss_prob > 0 and self.psd_weight > 0:
            raise ValueError("Set at most one of psd_loss_prob (stochastic) and psd_weight (deterministic)")
        return self


class WorldModelInferenceConfig(BaseModel):
    """Sampling knobs for the autoregressive denoising rollout."""

    n_diffusion_steps: int = 10
    # Noise level for the kv-cache update pass; None merges that pass into the
    # last diffusion step (one fewer forward per frame, see denoise_streaming).
    noise_level: float | None = 0.2
    # Sampling-schedule shape: "linear_quadratic" (default) or "linear" (uniform).
    schedule_type: str = "linear_quadratic"
    # Streaming KV-cache backing (B1). "grow" (default) rebuilds the per-layer cache with cat+clone
    # each frame; "ring" updates a fixed preallocated buffer in place (bit-exact, static shape --
    # infrastructure for CUDA-graph capture, E1). See docs/optimization_plan.md.
    streaming_cache: Literal["grow", "ring"] = "grow"
    # E1: capture the whole per-frame denoise (all diffusion steps + kv-update + ring rotation) into a
    # CUDA graph and replay it, eliminating per-kernel launch overhead. Requires a fixed cache
    # (forces "ring"); single-player, n_register_tokens==0 (PSD is supported, Tier B). Bit-exact
    # (verified maxdiff 0.0) and read only by the streaming `rollout()`, which AUTO-FALLS-BACK to eager
    # on any unsupported/failed capture (CPU, register tokens, etc.) -- so on-by-default is safe. The one
    # assumption is a single in-process rollout at a time (the static buffers are shared); concurrent
    # rollouts in one process must pass cuda_graphs=False. Default on: it's a free bit-exact speedup for
    # the interactive single-stream serving path (e.g. -8.5% at 4-step, and the 24.7ms 2-step PSD path).
    cuda_graphs: bool = True
