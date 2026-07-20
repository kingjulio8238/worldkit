"""B1 ring KV-cache: the in-place roll update must produce byte-identical buffers to the grow cat,
so the streaming rollout stays bit-exact."""

import torch


def _grow_update(k_ctx, new_k, nreg):
    return torch.cat([k_ctx[:, :nreg], k_ctx[:, nreg + 1 :], new_k], dim=1)


def _ring_update(buf, new, nreg):
    buf[:, nreg:].copy_(torch.roll(buf[:, nreg:], -1, dims=1))
    buf[:, -1:].copy_(new)
    return buf


def test_ring_update_matches_grow_no_register():
    torch.manual_seed(0)
    ctx = torch.randn(2, 6, 4, 8)  # (B, T_ctx, n_kv_heads, head_dim)
    new = torch.randn(2, 1, 4, 8)
    grown = _grow_update(ctx, new, nreg=0)
    ring = _ring_update(ctx.clone(), new, nreg=0)
    torch.testing.assert_close(ring, grown, atol=0, rtol=0)


def test_ring_update_matches_grow_with_register():
    torch.manual_seed(1)
    nreg = 2
    ctx = torch.randn(2, 2 + 5, 4, 8)  # nreg register slots + 5 context frames
    new = torch.randn(2, 1, 4, 8)
    grown = _grow_update(ctx, new, nreg=nreg)
    ring = _ring_update(ctx.clone(), new, nreg=nreg)
    torch.testing.assert_close(ring, grown, atol=0, rtol=0)
    # register slots must be untouched
    torch.testing.assert_close(ring[:, :nreg], ctx[:, :nreg], atol=0, rtol=0)


def test_ring_update_keeps_buffer_identity():
    # In-place update must keep the same tensor object (the CUDA-graph / static-address enabler).
    buf = torch.randn(1, 5, 4, 8)
    before = buf.data_ptr()
    _ring_update(buf, torch.randn(1, 1, 4, 8), nreg=0)
    assert buf.data_ptr() == before


def test_ring_multi_step_matches_grow():
    # Several sequential updates must stay identical between grow (new tensors) and ring (in place).
    torch.manual_seed(2)
    ctx0 = torch.randn(1, 6, 4, 8)
    grow = ctx0.clone()
    ring = ctx0.clone()
    for _ in range(10):
        new = torch.randn(1, 1, 4, 8)
        grow = _grow_update(grow, new, nreg=0)
        _ring_update(ring, new, nreg=0)
    torch.testing.assert_close(ring, grow, atol=0, rtol=0)
