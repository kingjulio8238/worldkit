"""Benchmark world-model INFERENCE on a random-init model: rollout latency, decode, end-to-end.

Random-init variant of :mod:`scripts.bench_wm_speed` (which needs a checkpoint). Speed is
weight-agnostic, so the numbers are the real ones; only output *quality* would differ from a trained
model (out of scope here -- for the quality-vs-speed tie-in run scripts/eval_world_model_offline.py
against a real checkpoint).

Reports, under bf16 autocast:
  * denoise rollout latency (ms/latent-frame, latent fps, x-realtime) over a sweep of diffusion steps;
  * codec decode throughput (video frames/s);
  * end-to-end (rollout + decode) latency.

    python scripts/bench_infer_speed.py --n-diffusion-steps 1 2 4 8 --compile
"""

from __future__ import annotations

import argparse
import json

import bench_lib as bl
import torch
from bench_optims import apply_optims, parse_optim_spec
from bench_quant import apply_quantization

from mira.inference.rollout import measure_rollout_speed, rollout
from mira.world_model.config import WorldModelInferenceConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="1b")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--n-diffusion-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--schedule-type", default="linear", choices=["linear", "linear_quadratic"])
    p.add_argument("--n-frames", type=int, default=16, help="latent frames to unroll")
    p.add_argument("--noise-level", type=lambda s: None if s.lower() == "none" else float(s), default=0.2)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--optim", default="", help="cast-reduction optims to apply, e.g. A1,A2,A3 or A4 (see bench_optims)")
    p.add_argument("--repeats", type=int, default=5, help="median over N timed rollouts per step count")
    p.add_argument("--streaming-cache", default="grow", choices=["grow", "ring"], help="B1 KV-cache backing")
    p.add_argument("--cuda-graphs", action="store_true", help="E1: whole-frame CUDA-graph replay")
    p.add_argument("--verify-graphs", action="store_true",
                   help="correctness check: graphed rollout vs eager (noise_level=0, fixed seed) -> maxdiff, then exit")
    p.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"],
                   help="E1/A6: reduce-overhead enables CUDA-graph trees; max-autotune is A6")
    p.add_argument("--quantize", default="none", choices=["none", "int8", "fp8"], help="C1/C2 weight-only quant")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl.enable_tf32(args.tf32)

    model = bl.build_random_world_model(size=args.size, device=device)
    model.eval()
    applied = apply_optims(model, parse_optim_spec(args.optim))
    quantized = apply_quantization(model, args.quantize)  # C: before compile (torchao+compile flow)
    if args.compile:
        # E1: "reduce-overhead" turns on CUDA-graph trees (captures the step, bypasses per-call dynamo
        # guards) -- the cheap first test of whether graphs fix the ring regression + cut launch idle.
        mode = None if args.compile_mode == "default" else args.compile_mode
        model.world_model.compile(mode=mode)
        model.decode_to_video = torch.compile(model.decode_to_video, mode=mode)

    td = model.temporal_downsampling
    latent_fps = model.config.video.fps / td

    if args.verify_graphs:
        # Correctness gate: the graphed rollout must match the eager rollout (deterministic at
        # noise_level=0 + fixed seed). Speed benches use random weights and can't catch a wiring bug.
        n_verify = min(6, args.n_frames)
        nvf = (model.n_context_latents + n_verify) * td
        steps = max(args.n_diffusion_steps)

        def _roll(cuda_graphs: bool):
            torch.manual_seed(1234)
            batch_v = bl.make_synthetic_batch(model, args.batch, device, n_video_frames=nvf)
            cfg = WorldModelInferenceConfig(
                n_diffusion_steps=steps, schedule_type=args.schedule_type, noise_level=0.0,
                streaming_cache="ring", cuda_graphs=cuda_graphs,
            )
            with torch.no_grad(), bl.autocast_ctx(device):
                return rollout(model, batch_v, cfg, n_frames=n_verify)

        z_graph = _roll(True)
        z_eager = _roll(False)
        maxdiff = (z_graph - z_eager).abs().max().item()
        rel = maxdiff / (z_eager.abs().max().item() + 1e-9)
        print(json.dumps({
            "verify_graphs": True, "n_diffusion_steps": steps, "n_frames": n_verify,
            "maxdiff": maxdiff, "rel": rel,
            "verdict": "BIT-EXACT" if maxdiff == 0 else ("CLOSE" if rel < 1e-3 else "MISMATCH"),
        }, indent=2))
        return
    # Size the clip in LATENT space so the stride-td codec encode yields exactly
    # (n_context_latents + n_frames) latents (n_context_frames can be odd, which would otherwise
    # drop a frame). n_iters in rollout is then exactly n_frames.
    n_video_frames = (model.n_context_latents + args.n_frames) * td
    batch = bl.make_synthetic_batch(model, args.batch, device, n_video_frames=n_video_frames)

    results: dict = {
        "size": args.size,
        "batch": args.batch,
        "compile": args.compile,
        "tf32": args.tf32,
        "optim": sorted(applied),
        "quantize": quantized,
        "streaming_cache": args.streaming_cache,
        "cuda_graphs": args.cuda_graphs,
        "compile_mode": args.compile_mode if args.compile else "eager",
        "latent_fps": round(latent_fps, 2),
        "rollout": [],
    }

    import statistics

    def mk_cfg(steps: int) -> WorldModelInferenceConfig:
        return WorldModelInferenceConfig(
            n_diffusion_steps=steps, schedule_type=args.schedule_type,
            noise_level=args.noise_level, streaming_cache=args.streaming_cache,
            cuda_graphs=args.cuda_graphs,
        )

    with torch.no_grad(), bl.autocast_ctx(device):
        for steps in args.n_diffusion_steps:
            cfg = mk_cfg(steps)
            # Warm up THIS step count's own graph (each n_diffusion_steps compiles a distinct graph;
            # warming only the max-step graph left the others paying a recompile on their first timed
            # frame -- the earlier ~5000 ms A3/A4 1-step spike).
            measure_rollout_speed(model, batch, cfg, n_frames=min(4, args.n_frames))
            # Repeat and take the median so a few-ms run-to-run jitter doesn't reorder the variants.
            samples = [
                measure_rollout_speed(model, batch, cfg, n_frames=args.n_frames)["denoise_ms_per_latent_frame"]
                for _ in range(args.repeats)
            ]
            ms = statistics.median(samples)
            s_per_frame = ms / 1000
            results["rollout"].append({
                "n_diffusion_steps": steps,
                "ms_per_latent_frame": round(ms, 1),
                "latent_fps": round(1 / s_per_frame, 2),
                "x_realtime": round((1 / s_per_frame) / latent_fps, 2),
                "repeats": args.repeats,
            })

        # Decode throughput: decode the unrolled latents back to video.
        z = rollout(model, batch, mk_cfg(min(args.n_diffusion_steps)), n_frames=args.n_frames)
        # Warm up decode first: with --compile the first call triggers Inductor compilation, and even
        # eager it pays cudnn autotune / kernel load. Timing the cold call would corrupt decode fps.
        model.decode_to_video(z)
        with bl.cuda_timer() as t:
            model.decode_to_video(z)
        n_latents = z.shape[1]
        video_frames = n_latents * td * args.batch
        results["decode"] = {
            "video_frames": video_frames,
            "decode_ms": round(t[0] * 1000, 1),
            "decode_video_fps": round(video_frames / t[0], 1),
        }

        # End-to-end: rollout (fewest steps) + decode.
        with bl.cuda_timer() as t:
            z = rollout(model, batch, mk_cfg(min(args.n_diffusion_steps)), n_frames=args.n_frames)
            model.decode_to_video(z)
        e2e_video_fps = (args.n_frames * td * args.batch) / t[0]
        results["end_to_end"] = {
            "n_diffusion_steps": min(args.n_diffusion_steps),
            "total_ms": round(t[0] * 1000, 1),
            "video_fps": round(e2e_video_fps, 2),
            "x_realtime": round(e2e_video_fps / model.config.video.fps, 2),
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
