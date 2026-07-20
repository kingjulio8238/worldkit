"""Toggle the inference cast-reduction optimizations (plan A1-A5) on a built model.

A1-A3 are now **real, config-flagged model code** (`qk_norm_fused`, `rope_upcast`, `rope_precompute`
on `LatentWorldModelConfig` / `SelfAttentionConfig`, implemented in `ml/attention.py` and
`world_model/layers/rope.py`). Because those flags are read at *forward* time, this module toggles
them on an already-built model (random-init or a loaded checkpoint) with no rebuild -- handy for the
bench and the quality gate. A5 (full bf16, deliberately NOT promoted) stays a bench-only module swap.

    A1  qk_norm_fused=True        -- fused F.layer_norm/F.rms_norm QK-norm
    A2  rope_upcast=False         -- bf16 RoPE (no fp32 round-trip)
    A3  rope precompute buffers   -- compile-friendly cos/sin tables
    A4  A1+A2+A3
    A5  bf16 QK-norm (no upcast) + bf16 RoPE -- fastest, quality-risky, bench-only

Design note (unchanged): the streaming KV-cache stores keys *after* QK-norm but *before* RoPE, so norm
and RoPE for keys can't be a single fused kernel without changing cache semantics. A4 is the reachable
optimum.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mira.ml.attention import QKLayerNorm, QKRMSNorm, SelfAttention
from mira.world_model.layers.rope import RoPE, SpatialRoPE2D

VALID = {"A1", "A2", "A3", "A4", "A5"}


def parse_optim_spec(spec: str) -> set[str]:
    """Parse a comma-separated --optim string into a normalized flag set (A4 expands to A1+A2+A3)."""
    if not spec:
        return set()
    flags = {s.strip().upper() for s in spec.split(",") if s.strip()}
    bad = flags - VALID
    if bad:
        raise ValueError(f"unknown --optim flags {sorted(bad)}; valid: {sorted(VALID)}")
    if "A4" in flags:
        flags |= {"A1", "A2", "A3"}
    return flags


class Bf16QKNorm(nn.Module):
    """A5 only: QK-norm computed directly in bf16 (no internal fp32 upcast). Fastest, quality-risky."""

    def __init__(self, original: QKLayerNorm | QKRMSNorm):
        super().__init__()
        self.qk_scale = original.qk_scale
        self.eps = original.eps
        self.is_rms = isinstance(original, QKRMSNorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_rms:
            var = x.pow(2).mean(-1, keepdim=True)
            normed = x * torch.rsqrt(var + self.eps)
        else:
            mean = x.mean(-1, keepdim=True)
            var = (x - mean).pow(2).mean(-1, keepdim=True)
            normed = (x - mean) * torch.rsqrt(var + self.eps)
        return normed * self.qk_scale.to(x.dtype)


def config_overrides_for(flags: set[str]) -> dict:
    """The LatentWorldModelConfig overrides that reproduce A1/A2/A3 at construction time (production
    path). A5 has no config equivalent (bench-only). Used when building a model from scratch."""
    ov: dict = {}
    if "A1" in flags:
        ov["qk_norm_fused"] = True
    if "A2" in flags or "A5" in flags:
        ov["rope_upcast"] = False
    if "A3" in flags:
        ov["rope_precompute"] = True
    return ov


def apply_optims(model, flags: set[str]) -> set[str]:
    """Toggle the selected optims on a built model (in place). Returns the flags applied.

    Call BEFORE ``world_model.compile()`` so the compiled graph captures the toggled paths. Drives the
    real config-flag code paths (A1-A3); A5 additionally swaps in the bench-only bf16 QK-norm.
    """
    if not flags:
        return flags

    inner = getattr(model, "single_world_model", model)  # unwrap the multiplayer wrapper if present
    dit = inner.world_model

    if "A1" in flags:  # real fused QK-norm branch
        for m in dit.modules():
            if isinstance(m, (QKLayerNorm, QKRMSNorm)):
                m.fused = True

    if "A2" in flags or "A5" in flags:  # bf16 RoPE (no fp32 upcast)
        for m in dit.modules():
            if isinstance(m, SelfAttention):
                m.rope_upcast = False

    if "A3" in flags:  # compile-friendly precomputed RoPE tables
        for m in dit.modules():
            if isinstance(m, RoPE):
                m.enable_precompute(dit.n_latent_frames + dit.n_register_tokens + 4)
            elif isinstance(m, SpatialRoPE2D):
                m.enable_precompute(dit.latent_height, dit.latent_width)

    if "A5" in flags:  # bench-only aggressive bf16 QK-norm
        for m in dit.modules():
            if isinstance(m, SelfAttention):
                m.q_ln = Bf16QKNorm(m.q_ln)
                m.k_ln = Bf16QKNorm(m.k_ln)

    return flags
