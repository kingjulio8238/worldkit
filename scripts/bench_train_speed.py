"""Benchmark world-model TRAINING throughput and MFU on a random-init model.

Mirrors :mod:`scripts.bench_wm_speed` (the inference latency bench) for the training side. Times a
real training step (forward + backward + optimizer step) under bf16 autocast and reports
samples/s, tokens/s, ms/step, peak memory and **MFU** (measured FLOPs, not a 6ND estimate).

Two modes isolate where the time goes:
    --mode dit   diffusion_loss on synthetic latents -- the trainable transformer only (default).
    --mode full  model(batch) end-to-end, including the frozen codec's DINOv3 encode.

Ablation knobs (compose them): --compile --tf32 --activation-ckpt --batch --size.

Single GPU:
    python scripts/bench_train_speed.py --mode dit --batch 1 --compile
Multi-GPU (DDP), launched with torchrun (the distributed setup reads LOCAL_RANK):
    torchrun --standalone --nproc_per_node 4 scripts/bench_train_speed.py --mode dit --batch 1

Optional dataloader-bound test (needs a real clip index; random-init runs skip it):
    python scripts/bench_train_speed.py --data-index /data/rl/test --dataloader-test
"""

from __future__ import annotations

import argparse
import json

import bench_lib as bl
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from mira.training.distributed import set_up_distributed
from mira.training.ema import ModelEMA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["dit", "full"], default="dit")
    p.add_argument("--size", default="1b")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=30, help="timed steps")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--activation-ckpt", action="store_true")
    p.add_argument("--peak-flops", type=float, default=bl.H100_SXM_BF16_PEAK_FLOPS)
    p.add_argument("--per-step-barrier", action="store_true", help="dist.barrier() every step (mirror trainer)")
    p.add_argument("--no-ema", action="store_true", help="skip the per-step EMA sweep (default: include it, as the trainer does)")
    # Optional real-dataloader test.
    p.add_argument("--data-index", default=None, help="clip index dir/json on the volume")
    p.add_argument("--dataloader-test", action="store_true")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--shuffle-buffer", type=int, default=100)
    return p.parse_args()


def _make_step_fn(args, model, device):
    """Return (step_fn, forward_fn, tokens_per_step). step_fn is a full fwd+bwd+opt(+EMA) step."""
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=torch.tensor(1e-4), betas=(0.9, 0.99), weight_decay=0.1,
    )

    # Mirror the trainer's per-step tail: a compiled optimizer step (the tensor lr lets it be
    # captured) and the EMA sweep over every trainable param -- a real, bandwidth-bound cost at 1B
    # params that the trainer pays each step (train_world_model.py:121-166).
    @torch.compile(disable=not args.compile)
    def optimizer_step():
        optimizer.step()

    ema = None if args.no_ema else ModelEMA(inner, decay=0.9999)

    dit = bl.unwrap(inner.world_model)  # world_model may itself be DDP-wrapped in dit mode
    tokens_per_step = args.batch * dit.n_latent_frames * dit.latent_height * dit.latent_width

    if args.mode == "dit":
        z, a = bl.make_synthetic_latents(inner, args.batch, device)

        def forward_fn():
            # diffusion_loss calls inner.world_model(...); when that is DDP-wrapped (dit + distributed)
            # the call goes through DDP.forward, so gradients ARE all-reduced on backward.
            with bl.autocast_ctx(device):
                return inner.diffusion_loss(z, a)["loss_total"]

    else:  # full
        batch = bl.make_synthetic_batch(inner, args.batch, device)

        def forward_fn():
            # DDP.__call__ forwards to the wrapped LatentWorldModel.forward, returning its loss dict.
            with bl.autocast_ctx(device):
                return model(batch)["loss_total"]

    def step_fn():
        optimizer.zero_grad(set_to_none=True)
        loss = forward_fn()
        loss.backward()
        optimizer_step()
        if ema is not None:
            ema.step()
        if args.per_step_barrier and dist.is_initialized():
            dist.barrier()

    return step_fn, forward_fn, tokens_per_step


def _measure_flops(args, model, device) -> dict[str, int]:
    """Measured forward FLOPs: the trainable DiT, and (full mode) the frozen codec encode."""
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    z, a = bl.make_synthetic_latents(inner, args.batch, device)

    def dit_fwd():
        with bl.autocast_ctx(device):
            inner.diffusion_loss(z, a)

    dit_fwd_flops = bl.measure_forward_flops(dit_fwd)
    out = {"dit_fwd_flops": dit_fwd_flops}

    if args.mode == "full":
        batch = bl.make_synthetic_batch(inner, args.batch, device)

        def codec_fwd():
            with torch.no_grad(), bl.autocast_ctx(device):
                inner.encode_video(batch.slice_time(0, inner.config.video.timesteps, fps=inner.config.video.fps))

        out["codec_fwd_flops"] = bl.measure_forward_flops(codec_fwd)
    return out


def run(args) -> dict:
    settings = set_up_distributed()
    device = torch.device(f"cuda:{settings.device}" if torch.cuda.is_available() else "cpu")
    is_main = settings.is_main_process
    is_distributed = dist.is_available() and dist.is_initialized()

    bl.enable_tf32(args.tf32)

    model = bl.build_random_world_model(size=args.size, device=device)
    model.train()
    if args.activation_ckpt:
        model.config.activation_checkpointing = True

    if args.compile:
        model.world_model.compile()
        if args.mode == "full":
            model.codec.encode = torch.compile(model.codec.encode)

    if is_distributed:
        if args.mode == "full":
            # Whole model through DDP.forward, exactly like the trainer.
            model = DistributedDataParallel(model, device_ids=[settings.device])
        else:
            # dit mode: diffusion_loss calls inner.world_model(...) directly, so wrap the DiT itself
            # (not the outer LWM) -- otherwise DDP.forward never runs and gradients are never
            # all-reduced, making multi-GPU dit numbers show zero comms overhead. (bos, ~5k params,
            # stays outside the reducer; negligible for a scaling measurement.)
            model.world_model = DistributedDataParallel(model.world_model, device_ids=[settings.device])

    step_fn, _, tokens_per_step = _make_step_fn(args, model, device)

    # FLOPs measured once, before the timed loop. Skipped when compiled (FlopCounterMode does not
    # compose with a compiled graph) or distributed (a DDP.forward with no matching backward would
    # desync the reducer; per-rank FLOPs equal the single-GPU run anyway -- read MFU from that).
    flop_info: dict[str, int] = {}
    if not args.compile and not is_distributed:
        try:
            flop_info = _measure_flops(args, model, device)
        except Exception as exc:  # noqa: BLE001 -- never let FLOP counting abort a paid run
            n_params = bl.count_trainable_params(model.module if isinstance(model, DistributedDataParallel) else model)
            flop_info = {"dit_fwd_flops": bl.analytic_forward_flops(n_params, tokens_per_step), "estimated": True}
            print(f"[warn] FLOP measurement failed ({type(exc).__name__}: {exc}); "
                  "using analytic 2*N*tokens estimate for MFU.", flush=True)
            # Release any fragments left by the failed forward so they don't cascade into the loop.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    bl.reset_peak_mem()
    # OOM is a legitimate "this config doesn't fit" result, not a harness failure: report it and
    # exit 0 so a batch sweep / ablation grid continues to the next config instead of aborting.
    try:
        stats = bl.time_loop(step_fn, args.steps, warmup=args.warmup)
    except torch.cuda.OutOfMemoryError:
        oom = {"mode": args.mode, "size": args.size, "batch_per_gpu": args.batch,
               "world_size": settings.world_size, "compile": args.compile, "tf32": args.tf32,
               "activation_ckpt": args.activation_ckpt, "status": "OOM",
               "hint": "retry with --activation-ckpt or a smaller --batch"}
        if is_main:
            print(json.dumps(oom, indent=2))
        if is_distributed:
            dist.destroy_process_group()
        return oom

    inner = model.module if isinstance(model, DistributedDataParallel) else model
    world_size = settings.world_size
    steps_per_sec = stats["iters_per_sec"]
    result = {
        "mode": args.mode,
        "size": args.size,
        "batch_per_gpu": args.batch,
        "world_size": world_size,
        "compile": args.compile,
        "tf32": args.tf32,
        "activation_ckpt": args.activation_ckpt,
        "ms_per_step": round(stats["ms_per_iter"], 2),
        "samples_per_sec_per_gpu": round(steps_per_sec * args.batch, 2),
        "samples_per_sec_global": round(steps_per_sec * args.batch * world_size, 2),
        "tokens_per_sec_per_gpu": round(steps_per_sec * tokens_per_step, 1),
        "peak_mem_gb": round(bl.peak_mem_gb(), 2),
        "trainable_params_M": round(bl.count_trainable_params(inner) / 1e6, 2),
    }
    if flop_info:
        dit_train_flops = bl.training_flops_per_step(flop_info["dit_fwd_flops"])
        result["dit_tflops_per_step"] = round(dit_train_flops / 1e12, 2)
        result["dit_mfu"] = round(bl.mfu(dit_train_flops, steps_per_sec, args.peak_flops), 4)
        if flop_info.get("estimated"):
            result["flops_estimated"] = True  # analytic 2*N*tokens fallback, not measured
        if "codec_fwd_flops" in flop_info:
            eff = dit_train_flops + flop_info["codec_fwd_flops"]
            result["codec_encode_tflops"] = round(flop_info["codec_fwd_flops"] / 1e12, 2)
            result["end_to_end_mfu"] = round(bl.mfu(eff, steps_per_sec, args.peak_flops), 4)

    if is_main:
        print(json.dumps(result, indent=2))
    if is_distributed:
        dist.destroy_process_group()
    return result


def dataloader_bound_test(args) -> dict:
    """Compare cached-batch throughput vs the real torchcodec loader to see if training is input-bound."""
    from mira.data.training_loader import create_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl.enable_tf32(args.tf32)
    model = bl.build_random_world_model(size=args.size, device=device)
    model.train()
    if args.compile:
        model.world_model.compile()
        model.codec.encode = torch.compile(model.codec.encode)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def train_on(batch):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with bl.autocast_ctx(device):
            loss = model(batch)["loss_total"]
        loss.backward()
        optimizer.step()

    # Cached-batch arm: one batch reused (no data movement).
    cached = bl.make_synthetic_batch(model, args.batch, device)
    cached_stats = bl.time_loop(lambda: train_on(cached), args.steps, warmup=args.warmup)

    # Real-loader arm.
    loader = create_loader(
        index_path=args.data_index,
        clip_len=model.config.video.timesteps,
        target_fps=model.config.video.fps,
        n_players=1,
        batch_size=args.batch,
        num_workers=args.num_workers,
        shuffle_buffer_size=args.shuffle_buffer,
        valid_keys=list(model.config.actions.valid_keys),
        action_fps=model.config.actions.target_fps,
        seed=0,
        infinite=True,
    )
    it = iter(loader)
    next(it)  # warm the workers

    def real_step():
        batch, _ = next(it)
        train_on(batch)

    real_stats = bl.time_loop(real_step, args.steps, warmup=args.warmup)

    cached_sps = cached_stats["iters_per_sec"] * args.batch
    real_sps = real_stats["iters_per_sec"] * args.batch
    out = {
        "cached_samples_per_sec": round(cached_sps, 2),
        "real_loader_samples_per_sec": round(real_sps, 2),
        "dataloader_overhead_pct": round(100 * (1 - real_sps / cached_sps), 1),
        "num_workers": args.num_workers,
        "verdict": "INPUT-BOUND" if real_sps < 0.9 * cached_sps else "GPU-BOUND",
    }
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    args = parse_args()
    if args.dataloader_test:
        if not args.data_index:
            raise SystemExit("--dataloader-test requires --data-index (a real clip index on the volume).")
        dataloader_bound_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
