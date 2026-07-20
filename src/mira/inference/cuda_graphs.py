"""E1: whole-frame CUDA-graph capture of the streaming denoise.

The per-frame denoise (all diffusion steps + kv-update + ring rotation, see
``LatentWorldModel._denoise_frame_body``) is launch-bound. This captures it into a single CUDA graph
and replays it per frame, eliminating per-kernel launch overhead. It requires everything the body
touches to live in **fixed buffers**: the ring KV-cache (B1) plus static input buffers here.

Manual capture (not ``torch.compile(mode="reduce-overhead")``, which can't handle the KV-cache being a
graph output). One graph is captured per ``(n_diffusion_steps, noise_level, schedule)``; the schedule
is baked in. Single-player, ``n_register_tokens==0`` (guarded). PSD models are supported: their extra
per-step ``tau_delta`` input is derived from the already-static ``delta_ts`` inside the captured body,
so no dynamic input crosses the graph boundary.

Everything is wrapped so any capture/replay failure **disables graphs and falls back to eager** — a
broken capture never crashes the run, it just loses the speedup. Assumes one rollout at a time (the
static buffers are shared), which is the interactive single-stream use case.
"""

from __future__ import annotations

import logging

import torch

from mira.world_model.schedule import build_inference_schedule

logger = logging.getLogger(__name__)


def _clone_cache(cache):
    return [None if x is None else (x[0].clone(), x[1].clone()) for x in cache]


def _copy_into(dst_cache, src_cache):
    for d, s in zip(dst_cache, src_cache):
        if d is not None:
            d[0].copy_(s[0])
            d[1].copy_(s[1])


class FrameGraphRunner:
    """Captures and replays one CUDA graph for the per-frame denoise body of a single-player model."""

    def __init__(self, model, n_diffusion_steps: int, noise_level: float | None, schedule_type: str):
        self.model = model
        self.n_steps = n_diffusion_steps
        self.noise_level = noise_level
        self.schedule_type = schedule_type
        self.graph: torch.cuda.CUDAGraph | None = None
        self.disabled = False
        # Static I/O buffers, allocated lazily from the first frame's real shapes.
        self.z_static = None
        self.a_static = None
        self.cp_static = None
        self.kv_static = None
        self.timesteps = None
        self.delta_ts = None

        cfg = model.config
        if cfg.n_register_tokens != 0:
            logger.warning("cuda_graphs unsupported (n_register_tokens!=0); falling back to eager.")
            self.disabled = True

    def _body(self) -> None:
        self.model._denoise_frame_body(
            self.z_static, self.a_static, self.cp_static, self.kv_static,
            self.timesteps, self.delta_ts, self.noise_level, ring_cache=True,
        )

    def _capture(self, z_cur, a_cur, clean_past, cache) -> None:
        device = z_cur.device
        self.timesteps = build_inference_schedule(self.n_steps, device, self.schedule_type)
        self.delta_ts = self.timesteps[1:] - self.timesteps[:-1]

        self.z_static = torch.empty_like(z_cur)
        self.a_static = torch.empty_like(a_cur)
        self.cp_static = torch.empty_like(clean_past) if clean_past is not None else None
        # The ring cache IS the static KV: adopt the real prefill values into fresh owned buffers.
        self.kv_static = _clone_cache(cache)
        saved = _clone_cache(cache)  # to restore after warmup/capture corrupt the buffers

        def reset():
            self.z_static.copy_(z_cur)
            self.a_static.copy_(a_cur)
            if self.cp_static is not None:
                self.cp_static.copy_(clean_past)
            _copy_into(self.kv_static, saved)

        # Warmup on a side stream (required before capture), re-seeding buffers each iter.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                reset()
                with torch.no_grad():
                    self._body()
        torch.cuda.current_stream().wait_stream(stream)

        reset()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                self._body()
        reset()  # leave the buffers holding the real frame-0 inputs for the first replay

    def run(self, z_cur, a_cur, clean_past, cache):
        """Denoise one frame via graph replay. Returns the (static) kv-cache, or None to signal the
        caller to run eager (graphs disabled / failed)."""
        if self.disabled:
            return None
        try:
            if self.graph is None:
                self._capture(z_cur, a_cur, clean_past, cache)  # clones this frame's prefill into kv_static
            elif cache is not self.kv_static:
                # A new rollout's first frame passes a fresh prefill cache (a different object); adopt
                # its context into the static buffers. Steady-state frames pass kv_static back (same
                # object) and skip this. Lets one captured graph serve every rollout (warmup + timed).
                _copy_into(self.kv_static, cache)
            self.z_static.copy_(z_cur)
            self.a_static.copy_(a_cur)
            if self.cp_static is not None:
                self.cp_static.copy_(clean_past)
            self.graph.replay()
            z_cur.copy_(self.z_static)  # write the denoised frame back into z_t
            return self.kv_static
        except Exception as exc:  # noqa: BLE001 -- never let a capture failure crash the rollout
            logger.warning("cuda_graphs capture/replay failed (%s: %s); falling back to eager.",
                           type(exc).__name__, exc)
            self.disabled = True
            return None
