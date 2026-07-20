"""Rotary position embeddings for the world model's temporal and spatial attention.

``RoPE`` is the time-axis embedding: positions are converted to seconds via ``grid_t / fps`` so the
frequencies are fps-dependent, with ``max_period_sec`` setting the slowest band. ``SpatialRoPE2D`` is
the axial 2D embedding for the height x width attention. Both are computed in fp32, and their values
are covered by a parity test.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class RoPE(nn.Module):
    """Inspired from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx.py#L179
    and https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/modules/rope.py"""

    w: Tensor  # registered buffer

    def __init__(
        self,
        dim: int,  # attention head dimension
        fps: int = 10,
        max_period_sec: float = 64.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.fps = fps
        self.max_period_sec = max_period_sec

        n_rope = 1
        self.dim_group = 2 * n_rope

        ds = torch.arange(self.dim // self.dim_group, dtype=torch.float32)
        self.register_buffer(
            "w", torch.exp(ds * (-math.log(self.max_period_sec) * 2 / self.dim)), persistent=False
        )  # (dim // 2)

    def _prepare_video_coords(
        self,
        num_frames: int,
        device: torch.device,
        offset: Tensor | None = None,
    ) -> torch.Tensor:
        # Always compute rope in fp32
        grid_t = torch.arange(num_frames, dtype=torch.float32, device=device)
        if offset is not None:
            grid_t = grid_t + offset.float()
        grid = torch.meshgrid(grid_t, indexing="ij")
        grid = torch.stack(grid, dim=0)  # (n_rope, T)  # n_rope = 1 if only doing time

        grid[0:1] = grid[0:1] / self.fps  # in seconds

        grid = grid.transpose(0, 1)  # (T, n_rope)  # n_rope = 1 if only doing time
        return grid

    def _compute(
        self, n_frames: int, device: torch.device, offset: Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = self._prepare_video_coords(n_frames, device=device, offset=offset)  # in seconds
        # (T, n_rope, dim // 2)
        freqs = self.w * grid.unsqueeze(-1)  # type: ignore
        freqs = freqs.flatten(1)  # (T, dim // 2)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)  # (T, dim)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % self.dim_group != 0:
            cos_padding = torch.ones_like(cos_freqs[:, : self.dim % self.dim_group])
            sin_padding = torch.zeros_like(cos_freqs[:, : self.dim % self.dim_group])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs

    def enable_precompute(self, max_frames: int, device: torch.device | None = None) -> None:
        """Precompute the cos/sin tables up to ``max_frames`` and slice per call (compile-friendly).

        The default path recomputes the tables on every forward; under ``torch.compile`` a runtime
        cache keyed on length triggers recompiles, whereas slicing a registered buffer does not. Only
        the ``offset is None`` path (the world-model hot path) is served from the buffer.
        """
        # Compute on the module's own device (buffers like ``w`` may already be on cuda when this is
        # enabled post-build); defaulting to CPU would mismatch them.
        cos, sin = self._compute(max_frames, device=device or self.w.device)
        self.register_buffer("_pre_cos", cos, persistent=False)
        self.register_buffer("_pre_sin", sin, persistent=False)

    def forward(
        self, n_frames: int, device: torch.device, offset: Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre_cos = getattr(self, "_pre_cos", None)
        if offset is None and pre_cos is not None and n_frames <= pre_cos.shape[0]:
            return pre_cos[:n_frames], self._pre_sin[:n_frames]
        return self._compute(n_frames, device=device, offset=offset)


class SpatialRoPE2D(nn.Module):
    """Axial 2D rotary position embedding for the spatial (height x width) attention.

    Positions are unnormalized token indices. This is useful because if we fine-tune a multi-player
    WM on top of a single player, we can just do a split-screen and it will work more or less out of
    the box, since the positions for player 1 are still valid.
    """

    inv_freq: Tensor  # registered buffer

    def __init__(self, dim: int, max_period: float = 100.0) -> None:
        super().__init__()
        assert dim % 4 == 0, f"2D RoPE needs head_dim divisible by 4, got {dim}"
        assert max_period >= 2, f"max_period is in tokens and must be >= 2 (Nyquist), got {max_period}"
        self.dim = dim
        self.max_period = max_period
        self.n_freqs = (dim // 2) // 2  # frequencies per spatial axis (x2 for repeat_interleave)

        # Log-spaced band: coarse end has a period of max_period,
        # the fine end has a period of 2 tokens
        inv_freq_min = 2 * math.pi / max_period
        inv_freq_max = math.pi
        k = torch.arange(self.n_freqs, dtype=torch.float32)
        inv_freq = inv_freq_min * (inv_freq_max / inv_freq_min) ** (k / max(self.n_freqs - 1, 1))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # (n_freqs,)

    def _axis_cos_sin(self, size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        coords = torch.arange(size, dtype=torch.float32, device=device)
        freqs = coords[:, None] * self.inv_freq[None, :]  # (size, n_freqs)
        cos = freqs.cos().repeat_interleave(2, dim=-1)  # (size, dim // 2)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return cos, sin

    def _compute(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cos_y, sin_y = self._axis_cos_sin(height, device)  # (height, dim // 2)
        cos_x, sin_x = self._axis_cos_sin(width, device)  # (width, dim // 2)

        # Broadcast the two axes onto the (h, w) grid and flatten as (h w) with y slowest.
        cos_y, sin_y = (t[:, None, :].expand(height, width, -1) for t in (cos_y, sin_y))
        cos_x, sin_x = (t[None, :, :].expand(height, width, -1) for t in (cos_x, sin_x))
        cos = torch.cat([cos_y, cos_x], dim=-1).reshape(height * width, self.dim)
        sin = torch.cat([sin_y, sin_x], dim=-1).reshape(height * width, self.dim)
        return cos, sin

    def enable_precompute(self, height: int, width: int, device: torch.device | None = None) -> None:
        """Precompute the (fixed) spatial cos/sin grid so forward returns a registered buffer, not a
        fresh recompute -- compile-friendly (see :meth:`RoPE.enable_precompute`)."""
        # Compute on the module's own device (see RoPE.enable_precompute); inv_freq may be on cuda.
        cos, sin = self._compute(height, width, device or self.inv_freq.device)
        self._pre_hw = (height, width)
        self.register_buffer("_pre_cos", cos, persistent=False)
        self.register_buffer("_pre_sin", sin, persistent=False)

    def forward(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "_pre_cos", None) is not None and getattr(self, "_pre_hw", None) == (height, width):
            return self._pre_cos, self._pre_sin
        return self._compute(height, width, device)
