"""Benchmark multiplayer player-count scaling: global O(p^2) vs tile_local O(p) spatial attention.

MIRA's multiplayer wrapper tiles ``p`` players into one grid (stacked along height) and runs one shared
transformer over it. With ``spatial_attention="global"`` the per-frame spatial self-attention is over
all ``p*h*w`` tokens -> O(p^2). ``spatial_attention="tile_local"`` runs block-diagonal attention within
each player's tile + a cheap cross-player mixer over pooled per-tile summaries -> O(p).

This builds a random-init model whose latent grid is ``p`` times taller (the codec runs per-player in
the real wrapper; the DiT sees the same tiled grid either way) and measures per-frame denoise latency
for each ``(mode, n_players)``, so the O(p^2) vs O(p) split shows up directly. Speed is weight-agnostic.

    python scripts/bench_multiplayer.py --n-players 1 2 4 8 16 --compile
"""

from __future__ import annotations

import argparse
import json
import statistics

import bench_lib as bl
import torch

from mira.world_model import latent_world_model as lwm
from mira.world_model.config import WorldModelInferenceConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-players", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--modes", nargs="+", default=["global", "tile_local"],
                   choices=["global", "tile_local"])
    p.add_argument("--size", default="1b")
    p.add_argument("--n-diffusion-steps", type=int, default=4)
    p.add_argument("--n-frames", type=int, default=2, help="latent frames to unroll per measurement")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tf32", action="store_true")
    return p.parse_args()


def _measure(model, args, device) -> dict:
    """Per-frame denoise latency for one (already-built) model."""
    from mira.inference.rollout import measure_rollout_speed  # noqa: PLC0415

    td = model.temporal_downsampling
    n_video_frames = (model.n_context_latents + args.n_frames) * td
    batch = bl.make_synthetic_batch(model, 1, device, n_video_frames=n_video_frames)
    cfg = WorldModelInferenceConfig(
        n_diffusion_steps=args.n_diffusion_steps, schedule_type="linear", noise_level=0.2,
        streaming_cache="grow", cuda_graphs=False,  # isolate the attention scaling from graph capture
    )
    warm = min(2, args.n_frames)
    with torch.no_grad(), bl.autocast_ctx(device):
        measure_rollout_speed(model, batch, cfg, n_frames=warm)  # warm compile/cudnn
        samples = [
            measure_rollout_speed(model, batch, cfg, n_frames=args.n_frames)["denoise_ms_per_latent_frame"]
            for _ in range(args.repeats)
        ]
    return {"ms": round(statistics.median(samples), 2)}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl.enable_tf32(args.tf32)

    node = bl._compose_world_model_config(args.size, multiplayer=False)
    base_height = lwm._config_dict_from_yaml(node)["video"]["height"]

    results: dict[str, dict[int, float | None]] = {m: {} for m in args.modes}
    grid: dict[int, str] = {}
    for p in args.n_players:
        for mode in args.modes:
            print(f"\n===== mode={mode} n_players={p} (video_height={base_height * p}) =====", flush=True)
            model = None
            try:
                model = bl.build_random_world_model(
                    size=args.size, device=device, video_height=base_height * p,
                    wm_config_overrides={"spatial_attention": mode, "n_spatial_tiles": p},
                )
                model.eval()
                if args.compile:
                    model.world_model.compile()
                out = _measure(model, args, device)
                results[mode][p] = out["ms"]
                lh, lw = model.world_model.latent_height, model.world_model.latent_width
                grid[p] = f"{lh}x{lw}={lh * lw} tok/frame"
                print(json.dumps({"mode": mode, "n_players": p, **out, "grid": grid[p]}), flush=True)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # noqa: PERF203
                results[mode][p] = None
                print(f"  OOM/err at mode={mode} p={p}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # Report: latency + scaling relative to the smallest player count run per mode.
    print("\n\n===== player-count scaling (ms/frame @ "
          f"{args.n_diffusion_steps} steps, compile={args.compile}) =====")
    header = f"{'p':>4} {'grid':>18} " + " ".join(f"{m:>12}" for m in args.modes) \
             + " " + " ".join(f"{m + '/p1':>10}" for m in args.modes)
    print(header)
    p0 = args.n_players[0]
    for p in args.n_players:
        cells = []
        for m in args.modes:
            v = results[m].get(p)
            cells.append(f"{v:>12.2f}" if v is not None else f"{'OOM':>12}")
        ratios = []
        for m in args.modes:
            v, v0 = results[m].get(p), results[m].get(p0)
            ratios.append(f"{v / v0:>10.1f}" if (v and v0) else f"{'-':>10}")
        print(f"{p:>4} {grid.get(p, '?'):>18} " + " ".join(cells) + " " + " ".join(ratios))

    # Empirical scaling exponent (log-log slope) between the first and last successful player counts.
    import math
    print("\nscaling exponent (ms ~ p^k, k=2 -> quadratic, k=1 -> linear):")
    for m in args.modes:
        pts = [(p, results[m][p]) for p in args.n_players if results[m].get(p)]
        if len(pts) >= 2:
            (p_lo, v_lo), (p_hi, v_hi) = pts[0], pts[-1]
            k = math.log(v_hi / v_lo) / math.log(p_hi / p_lo) if p_hi > p_lo else float("nan")
            print(f"  {m:>12}: k = {k:.2f}  ({p_lo}->{p_hi} players)")


if __name__ == "__main__":
    main()
