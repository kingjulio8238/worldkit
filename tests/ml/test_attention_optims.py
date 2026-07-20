"""Tests for the opt-in inference cast-reduction optimizations A1 (fused QK-norm), A2 (bf16 RoPE),
and A3 (precomputed RoPE tables). A1 and A3 must be equivalence-preserving; A2 is a bounded bf16
approximation.
"""

import torch

from mira.ml.attention import (
    QKLayerNorm,
    QKRMSNorm,
    SelfAttention,
    SelfAttentionConfig,
    apply_rotary_emb,
)
from mira.world_model.layers.rope import RoPE, SpatialRoPE2D

torch.manual_seed(0)


def _rand_qk(dtype=torch.float32):
    # (batch, time, n_head, head_dim) as QK-norm sees it inside attention.
    return torch.randn(2, 5, 4, 32, dtype=dtype)


def test_a1_fused_layernorm_matches_eager_fp32():
    x = _rand_qk()
    ref = QKLayerNorm((4, 32))
    fused = QKLayerNorm((4, 32), fused=True)
    fused.load_state_dict(ref.state_dict())
    # randomize the affine so it isn't the identity
    with torch.no_grad():
        ref.qk_scale.normal_()
        fused.qk_scale.copy_(ref.qk_scale)
    torch.testing.assert_close(fused(x), ref(x), atol=1e-5, rtol=1e-4)


def test_a1_fused_rmsnorm_matches_eager_fp32():
    x = _rand_qk()
    ref = QKRMSNorm((4, 32))
    fused = QKRMSNorm((4, 32), fused=True)
    with torch.no_grad():
        ref.qk_scale.normal_()
        fused.qk_scale.copy_(ref.qk_scale)
    torch.testing.assert_close(fused(x), ref(x), atol=1e-5, rtol=1e-4)


def test_a1_fused_layernorm_bf16_bounded():
    x = _rand_qk(torch.bfloat16)
    ref = QKLayerNorm((4, 32))
    fused = QKLayerNorm((4, 32), fused=True)
    # bf16 in, both upcast internally -> close but not bit-exact.
    torch.testing.assert_close(fused(x).float(), ref(x).float(), atol=2e-2, rtol=2e-2)


def _rope_freqs(t: int, dim: int):
    cos = torch.randn(t, dim)
    sin = torch.randn(t, dim)
    return cos, sin


def test_a2_rope_upcast_true_is_unchanged():
    # upcast=True (default) must reproduce the released fp32 path bit-for-bit.
    x = torch.randn(2, 6, 4, 16)
    freqs = _rope_freqs(6, 16)
    a = apply_rotary_emb(x, freqs)  # default upcast=True
    b = apply_rotary_emb(x, freqs, upcast=True)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_a2_rope_bf16_matches_fp32_within_tol():
    x = torch.randn(2, 6, 4, 16, dtype=torch.bfloat16)
    freqs = _rope_freqs(6, 16)  # fp32 cos/sin, as produced by RoPE
    up = apply_rotary_emb(x, freqs, upcast=True)
    bf = apply_rotary_emb(x, freqs, upcast=False)
    assert bf.dtype == x.dtype
    torch.testing.assert_close(bf.float(), up.float(), atol=5e-2, rtol=5e-2)


def test_a3_rope_precompute_exact():
    rope = RoPE(dim=64)
    dev = torch.device("cpu")
    rope.enable_precompute(max_frames=32, device=dev)
    for n in (1, 8, 20, 32):
        pc, ps = rope(n, dev)
        cc, cs = rope._compute(n, dev)
        torch.testing.assert_close(pc, cc, atol=0, rtol=0)
        torch.testing.assert_close(ps, cs, atol=0, rtol=0)


def test_a3_rope_precompute_falls_back_beyond_max():
    rope = RoPE(dim=64)
    dev = torch.device("cpu")
    rope.enable_precompute(max_frames=16, device=dev)
    # n beyond the precomputed length must still be correct (recompute path).
    pc, ps = rope(40, dev)
    cc, cs = rope._compute(40, dev)
    torch.testing.assert_close(pc, cc, atol=0, rtol=0)


def test_a3_precompute_uses_module_device_not_cpu():
    # Regression: enable_precompute() with no device arg must compute on the module's own device
    # (its buffers may already be on cuda when enabled post-build), not hardcode CPU.
    rope = RoPE(dim=64)
    rope.enable_precompute(max_frames=8)
    assert rope._pre_cos.device == rope.w.device
    srope = SpatialRoPE2D(dim=64)
    srope.enable_precompute(4, 4)
    assert srope._pre_cos.device == srope.inv_freq.device


def test_a3_spatial_rope_precompute_exact():
    srope = SpatialRoPE2D(dim=64)
    dev = torch.device("cpu")
    srope.enable_precompute(9, 16, device=dev)
    pc, ps = srope(9, 16, dev)
    cc, cs = srope._compute(9, 16, dev)
    torch.testing.assert_close(pc, cc, atol=0, rtol=0)
    torch.testing.assert_close(ps, cs, atol=0, rtol=0)
    # a different grid falls back to recompute (correct).
    pc2, _ = srope(4, 4, dev)
    cc2, _ = srope._compute(4, 4, dev)
    torch.testing.assert_close(pc2, cc2, atol=0, rtol=0)


def test_config_flags_flow_to_selfattention():
    cfg = SelfAttentionConfig(
        embed_dim=128, num_heads=4, num_kv_heads=2, qk_norm_fused=True, rope_upcast=False
    )
    attn = SelfAttention(cfg)
    assert attn.q_ln.fused and attn.k_ln.fused
    assert attn.rope_upcast is False
    # defaults preserve released behaviour
    default = SelfAttention(SelfAttentionConfig(embed_dim=128, num_heads=4, num_kv_heads=2))
    assert not default.q_ln.fused
    assert default.rope_upcast is True
