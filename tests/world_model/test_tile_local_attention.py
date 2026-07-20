"""Tile-local (O(p)) spatial attention for multiplayer scaling.

Checks: global mode is untouched by the tile count; tile_local produces the right shape and differs
from global; and the structural guarantee that makes it O(p) -- per-player block-diagonal attention +
a permutation-symmetric cross-player mixer -- shows up as player-permutation equivariance.
"""

import torch
from einops import rearrange

from mira.ml.attention import SelfAttentionConfig
from mira.world_model.layers.rope import SpatialRoPE2D
from mira.world_model.layers.transformer import AdaSTBlock

C, HEADS = 64, 8
HEAD_DIM = C // HEADS


def _block(mode, p, seed=0):
    torch.manual_seed(seed)
    cfg = SelfAttentionConfig(embed_dim=C, num_heads=HEADS, num_kv_heads=HEADS)
    return AdaSTBlock(cfg, cond_dim=C, causal=False, time_attention=False, ada_attn_ln=True,
                      spatial_attention=mode, n_spatial_tiles=p)


def _inputs(p, hs=4, w=5, b=2, t=2, seed=1):
    torch.manual_seed(seed)
    h = p * hs
    x = torch.randn(b, t, h, w, C)
    cond = torch.randn(b, t, h, w, C)
    sp = SpatialRoPE2D(dim=HEAD_DIM)(h, w, torch.device("cpu"))
    return x, cond, sp


def test_shapes_both_modes():
    p = 3
    x, cond, sp = _inputs(p)
    for mode in ("global", "tile_local"):
        y, _ = _block(mode, p)(x, cond, spatial_rotary_emb=sp)
        assert y.shape == x.shape


def test_global_ignores_tile_count():
    # In global mode the tile count is inert -> identical output to a single-grid model.
    p = 4
    x, cond, sp = _inputs(p)
    y_p, _ = _block("global", p)(x, cond, spatial_rotary_emb=sp)
    y_1, _ = _block("global", 1)(x, cond, spatial_rotary_emb=sp)
    torch.testing.assert_close(y_p, y_1, atol=0, rtol=0)


def test_tile_local_differs_from_global():
    p = 3
    x, cond, sp = _inputs(p)
    y_g, _ = _block("global", p)(x, cond, spatial_rotary_emb=sp)
    y_t, _ = _block("tile_local", p)(x, cond, spatial_rotary_emb=sp)
    assert not torch.allclose(y_g, y_t)


def test_tile_local_player_permutation_equivariance():
    """Permuting the player tiles permutes the output tiles -- the O(p) block-diagonal + symmetric-mixer
    structure. (Would fail for global attention, which couples all tiles asymmetrically via RoPE.)"""
    p, hs, w = 3, 4, 5
    x, cond, sp = _inputs(p, hs=hs, w=w)
    blk = _block("tile_local", p).eval()
    with torch.no_grad():
        y, _ = blk(x, cond, spatial_rotary_emb=sp)

        perm = torch.tensor([2, 0, 1])
        def permute_tiles(z):
            z = rearrange(z, "b t (p hs) w c -> b t p hs w c", p=p)[:, :, perm]
            return rearrange(z, "b t p hs w c -> b t (p hs) w c")

        y_from_permuted, _ = blk(permute_tiles(x), permute_tiles(cond), spatial_rotary_emb=sp)
    torch.testing.assert_close(y_from_permuted, permute_tiles(y), atol=1e-5, rtol=1e-4)
