"""The spatio-temporal transformer block (``AdaSTBlock``) used by the diffusion transformer.

Each block runs spatial self-attention over the ``(h w)`` tokens of every frame, optional temporal
self-attention over the ``t`` axis of every spatial cell, and a gated feed-forward, all conditioned
through adaptive LayerNorm. The attention primitives are reused from :mod:`mira.ml.attention`.
"""

from __future__ import annotations

import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor, nn

from mira.ml.attention import AdaptiveLayerNorm, SelfAttention, SelfAttentionConfig


class FeedForward(nn.Module):
    """SwiGLU feed-forward with a hidden width rounded up to a multiple of ``multiple_of``."""

    def __init__(self, dim: int, dim_multiplier: int = 4, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * dim_multiplier * dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.swish_linear = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_linear = nn.Linear(dim, hidden_dim, bias=False)
        self.output_linear = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_linear(F.silu(self.swish_linear(x)) * self.gate_linear(x))


class AdaSTBlock(nn.Module):
    """Spatial attention + optional temporal attention + feed-forward, with AdaLN conditioning.

    Args:
        config: Self-attention configuration shared by the spatial and temporal attention.
        cond_dim: Dimension of the conditioning tensor fed to the adaptive LayerNorms.
        causal: Whether the temporal attention is causal.
        time_attention: Whether this block has a temporal-attention sublayer.
        ada_attn_ln: If True, apply AdaLN conditioning to the attention sublayers (space/time) as
            well as the MLP, as in standard DiT; otherwise the attention LayerNorms are plain.
    """

    def __init__(
        self,
        config: SelfAttentionConfig,
        cond_dim: int,
        causal: bool,
        time_attention: bool = True,
        ada_attn_ln: bool = False,
        spatial_attention: str = "global",
        n_spatial_tiles: int = 1,
    ):
        super().__init__()
        self.time_attention = time_attention
        self.ada_attn_ln = ada_attn_ln
        self.spatial_attention = spatial_attention
        self.n_spatial_tiles = n_spatial_tiles

        self.space_attn_ln = (
            AdaptiveLayerNorm(config.embed_dim, cond_dim) if ada_attn_ln else nn.LayerNorm(config.embed_dim)
        )
        self.space_attn = SelfAttention(config, causal=False)

        # tile_local: a cheap cross-player mixer so per-tile (block-diagonal) attention still lets
        # players see each other. It attends over just the p pooled per-tile summaries (O(p) work vs the
        # O(p^2) of full spatial attention), then broadcasts the mixed summary back into each tile.
        self.player_mix_ln, self.player_attn = None, None
        if spatial_attention == "tile_local":
            self.player_mix_ln = nn.LayerNorm(config.embed_dim)
            self.player_attn = SelfAttention(config, causal=False)

        self.time_attn_ln, self.time_attn = None, None
        if self.time_attention:
            self.time_attn_ln = (
                AdaptiveLayerNorm(config.embed_dim, cond_dim)
                if ada_attn_ln
                else nn.LayerNorm(config.embed_dim)
            )
            self.time_attn = SelfAttention(config, causal=causal)

        self.mlp_ln = AdaptiveLayerNorm(config.embed_dim, cond_dim)
        self.mlp = FeedForward(config.embed_dim)

    def _tile_local_space(self, x, cond_space, spatial_rotary_emb, h, w):
        """O(p) spatial attention: block-diagonal per player tile + a pooled cross-player mixer.

        ``x``/``cond_space`` are ``(b t) (h w) c`` with ``h = n_spatial_tiles * hs`` (players stacked
        along height). We run spatial attention independently within each tile (batching the tile axis),
        then mix the p pooled per-tile summaries so players still exchange information.
        """
        p = self.n_spatial_tiles
        hs = h // p
        xs = rearrange(x, "bt (p hs w) c -> (bt p) (hs w) c", p=p, hs=hs, w=w)
        # RoPE is relative, so tile 0's slice encodes the same within-tile geometry as every tile.
        tile_rope = None
        if spatial_rotary_emb is not None:
            tile_rope = (spatial_rotary_emb[0][: hs * w], spatial_rotary_emb[1][: hs * w])
        if self.ada_attn_ln:
            cs = rearrange(cond_space, "bt (p hs w) c -> (bt p) (hs w) c", p=p, hs=hs, w=w)
            normed = self.space_attn_ln(xs, cs)
        else:
            normed = self.space_attn_ln(xs)
        xs = xs + self.space_attn(normed, rotary_emb=tile_rope)

        # Cross-player mixer over the p per-tile summaries (tiny: p tokens), broadcast back into tiles.
        summary = reduce(xs, "(bt p) sw c -> bt p c", "mean", p=p)
        mixed = self.player_attn(self.player_mix_ln(summary))
        xs = xs + rearrange(mixed, "bt p c -> (bt p) 1 c")
        return rearrange(xs, "(bt p) (hs w) c -> bt (p hs w) c", p=p, hs=hs, w=w)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        temporal_rotary_emb: Tensor | None = None,
        spatial_rotary_emb: Tensor | None = None,
        return_kv: bool = False,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        b, t, h, w, _ = x.shape
        x = rearrange(x, "b t h w c -> (b t) (h w) c")
        cond_space = rearrange(cond, "b t h w c -> (b t) (h w) c") if self.ada_attn_ln else None
        if self.spatial_attention == "tile_local" and self.n_spatial_tiles > 1:
            x = self._tile_local_space(x, cond_space, spatial_rotary_emb, h, w)
        else:
            normed = self.space_attn_ln(x, cond_space) if self.ada_attn_ln else self.space_attn_ln(x)
            x = x + self.space_attn(normed, rotary_emb=spatial_rotary_emb)

        to_cache = None
        if self.time_attn is not None:
            assert self.time_attn_ln is not None
            x = rearrange(x, "(b t) (h w) c -> (b h w) t c", b=b, t=t, h=h, w=w)
            if self.ada_attn_ln:
                cond_time = rearrange(cond, "b t h w c -> (b h w) t c")
                x_normed = self.time_attn_ln(x, cond_time)
            else:
                x_normed = self.time_attn_ln(x)
            y_out = self.time_attn(
                x_normed,
                rotary_emb=temporal_rotary_emb,
                return_kv=return_kv,
                kv_cache=kv_cache,
            )  # type: ignore
            if return_kv:
                y, to_cache = y_out
            else:
                y = y_out
            x = x + y
            x = rearrange(x, "(b h w) t c -> b t h w c", b=b, h=h, w=w)
        else:
            x = rearrange(x, "(b t) (h w) c -> b t h w c", b=b, t=t, h=h, w=w)

        x = x + self.mlp(self.mlp_ln(x, cond))

        return x, to_cache
