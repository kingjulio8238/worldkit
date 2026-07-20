"""Profile one training (or inference) step to attribute time to kernels and spot stalls.

Uses ``torch.profiler`` to record a few steps and print the top CUDA kernels by self-time, plus a
CPU/GPU breakdown that surfaces host-side stalls (dataloader waits, sync points) and the fp32
QK-norm / RoPE cast overhead. Optionally writes a Chrome trace for Perfetto/Nsight inspection.

    python scripts/bench_profile.py --mode dit --compile
    python scripts/bench_profile.py --mode infer --trace-out /data/trace.json
"""

from __future__ import annotations

import argparse

import bench_lib as bl
import torch
from bench_optims import apply_optims, parse_optim_spec
from torch.profiler import ProfilerActivity, profile, schedule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["dit", "full", "infer"], default="dit")
    p.add_argument("--size", default="1b")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--n-frames", type=int, default=8, help="infer mode: latent frames to slide over")
    p.add_argument("--n-diffusion-steps", type=int, default=4, help="infer mode: denoise steps per frame")
    p.add_argument("--trace-out", default=None, help="write a Chrome trace here")
    p.add_argument("--rows", type=int, default=25)
    p.add_argument("--optim", default="", help="cast-reduction optims to apply, e.g. A1,A2,A3 or A4")
    p.add_argument("--streaming-cache", default="grow", choices=["grow", "ring"], help="infer mode: B1 KV-cache backing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl.enable_tf32(args.tf32)

    model = bl.build_random_world_model(size=args.size, device=device)
    apply_optims(model, parse_optim_spec(args.optim))
    if args.compile:
        model.world_model.compile()

    if args.mode == "infer":
        # Profile the STEADY-STATE per-frame streaming denoise -- the interactive latency the design
        # optimizes -- not the whole rollout (whose one-time DINO context encode, ~as costly as all
        # the denoising, would swamp the signal). The one-time context prefill is done once, outside
        # the profiled region. Reuses rollout's own action-offset helper so there is no logic drift.
        from mira.inference.rollout import _encode_window_actions, _inner_model  # noqa: PLC0415

        model.eval()
        inner = _inner_model(model)
        td = inner.temporal_downsampling
        n_ctx = inner.n_context_latents
        window = n_ctx + 1
        n_steps = args.n_diffusion_steps
        ring = args.streaming_cache == "ring"
        nvf = (n_ctx + max(2, args.n_frames)) * td
        batch = bl.make_synthetic_batch(model, args.batch, device, n_video_frames=nvf)
        inner.codec.preprocess_batch(batch)
        batch = batch.to(device)
        with torch.no_grad(), bl.autocast_ctx(device):
            z = inner.encode_video(batch).clone()
        z_t = torch.randn_like(z)
        z_t[:, :n_ctx] = z[:, :n_ctx]
        n_windows = z.shape[1] - window + 1
        # One-time context prefill (fills the streaming kv-cache); NOT profiled.
        with torch.no_grad(), bl.autocast_ctx(device):
            a0 = _encode_window_actions(model, inner, batch, 0, window)
            _, kv = inner.denoise_streaming(
                z_t[:, 0:window].clone(), a0, streaming_kv_caches=None,
                n_diffusion_steps=n_steps, noise_level=0.2, schedule_type="linear", ring_cache=ring,
            )
        state = {"i": 1, "kv": kv}

        def step():
            s = state["i"] % n_windows  # slide the window; cache carries the context
            state["i"] += 1
            with torch.no_grad(), bl.autocast_ctx(device):
                a = _encode_window_actions(model, inner, batch, s, window)
                _, state["kv"] = inner.denoise_streaming(
                    z_t[:, s:s + window].clone(), a, streaming_kv_caches=state["kv"],
                    n_diffusion_steps=n_steps, noise_level=0.2, schedule_type="linear", ring_cache=ring,
                )
    else:
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        if args.mode == "dit":
            z, a = bl.make_synthetic_latents(model, args.batch, device)

            def fwd():
                with bl.autocast_ctx(device):
                    return model.diffusion_loss(z, a)["loss_total"]
        else:
            batch = bl.make_synthetic_batch(model, args.batch, device)

            def fwd():
                with bl.autocast_ctx(device):
                    return model(batch)["loss_total"]

        def step():
            optimizer.zero_grad(set_to_none=True)
            fwd().backward()
            optimizer.step()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # wait/warmup/active schedule so compile + autotune settle before the recorded window.
    sched = schedule(wait=1, warmup=3, active=3, repeat=1)
    with profile(activities=activities, schedule=sched, record_shapes=False, with_stack=False) as prof:
        for _ in range(7):
            step()
            prof.step()

    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=args.rows))
    if args.trace_out:
        prof.export_chrome_trace(args.trace_out)
        print(f"\nChrome trace written to {args.trace_out}")


if __name__ == "__main__":
    main()
