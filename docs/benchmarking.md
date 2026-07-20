# Benchmarking training & inference (Modal / H100)

This harness answers **"is mira's training and inference optimized?"** with numbers, on a random-init
model built from the repo's own configs — no checkpoint needed. Speed, MFU and latency are
weight-agnostic; only output *quality* would differ from a trained model (out of scope here — for the
quality-vs-speed tie-in, run `scripts/eval_world_model_offline.py` against a real checkpoint).

## What's here

| File | Role |
|---|---|
| `scripts/bench_lib.py` | Shared: random-init model builder, synthetic data, FLOP/MFU, timing helpers |
| `scripts/bench_train_speed.py` | Training throughput + MFU (`--mode dit`/`full`), dataloader-bound test |
| `scripts/bench_infer_speed.py` | Rollout latency + decode throughput + end-to-end (random-init variant of `bench_wm_speed.py`) |
| `scripts/bench_profile.py` | `torch.profiler` on one step → top kernels + Chrome trace |
| `scripts/modal_bench.py` | Modal H100 entrypoints wrapping all of the above |

**Random-init detail:** `LatentWorldModel.__init__` requires a frozen codec checkpoint.
`bench_lib.build_random_world_model` builds a random `VideoCodec` instead (DINOv3 via `torch.hub` with
`pretrained=False` — code only, no gated weights) and injects it through the codec-load hook, so the
real constructor runs unchanged. Validated on CPU: 1.19B trainable params, forward FLOPs measured at
6.75 TFLOP (≈ 2·N·tokens, matching the analytic estimate).

## Modal commands (the handoff)

Prereq: `modal token new` (or the workspace the sibling repos use). All functions run on H100 and
**cost money** — the image bakes torch cu128 + torchcodec + the repo + DINOv3, so the first run pays a
one-time build.

```bash
# --- Phase 1: inference latency (compile off vs on) ---
modal run scripts/modal_bench.py::infer                        # steps 1 2 4 8, compile off
modal run scripts/modal_bench.py::infer --compile              # + torch.compile
modal run scripts/modal_bench.py::infer --compile --tf32       # + TF32

# --- Phase 2: training throughput + MFU ---
modal run scripts/modal_bench.py::train --mode dit             # trainable transformer only
modal run scripts/modal_bench.py::train --mode dit --compile
modal run scripts/modal_bench.py::train --mode full            # end-to-end incl. codec DINO encode

# --- Phase 3: profiling (writes a Chrome trace to the volume) ---
modal run scripts/modal_bench.py::profile --mode dit
modal run scripts/modal_bench.py::profile --mode dit --compile

# --- Phase 4: ablation grid + DDP scaling ---
modal run scripts/modal_bench.py::ablation --mode dit          # baseline→+compile→+tf32→+actckpt
modal run scripts/modal_bench.py::ddp --mode dit               # 1 / 2 / 4 H100, samples/s each

# --- optional: dataloader-bound test (needs real data on the volume) ---
modal volume put mira-bench-data /local/rl/test /rl/test
modal run scripts/modal_bench.py::dataloader_test --data-index /data/rl/test

# convenience: core matrix in one shot
modal run scripts/modal_bench.py
```

Run a bench directly (e.g. on your own GPU box) without Modal:

```bash
cd scripts
python bench_infer_speed.py --n-diffusion-steps 1 2 4 8 --compile
python bench_train_speed.py --mode dit --compile
torchrun --standalone --nproc_per_node 4 bench_train_speed.py --mode dit --per-step-barrier
```

## How to read the numbers

**Inference (`infer`)** emits per-step-count `ms_per_latent_frame`, `latent_fps`, `x_realtime`, plus
`decode` (video fps) and `end_to_end` (rollout+decode, `x_realtime` vs the 20 fps video rate).
- Verdict = the **min `n_diffusion_steps` that stays ≥ 1× realtime** end-to-end.
- Compare compile-off vs compile-on: the delta is what Inductor buys inference.

**Training (`train`)** emits JSON with `samples_per_sec`, `tokens_per_sec`, `ms_per_step`,
`peak_mem_gb`, `dit_tflops_per_step`, and **`dit_mfu`** (the headline: trainable-transformer MFU vs
H100 bf16 peak). `full` mode adds `codec_encode_tflops` + `end_to_end_mfu`.
- **MFU < ~35–40%** on a dense transformer = real headroom (kernels, launch overhead, or input-bound).
- `dit` vs `full` gap = how much the frozen DINO encode adds to each training step.
- The timed step is a faithful `fwd + bwd + compiled-optimizer-step + EMA` (the EMA sweep over all
  ~1B params is a real per-step cost the trainer pays; `--no-ema` drops it). The LR scheduler step is
  omitted (negligible, touches no params).
- **MFU is reported only on eager, single-GPU runs.** It is skipped under `--compile` (FlopCounterMode
  can't trace a compiled graph) and under DDP (per-rank FLOPs equal the 1-GPU run) — read MFU from the
  plain `train --mode dit` run and read throughput/scaling from the compiled / DDP runs.

**Profiling (`profile`)** prints top CUDA kernels by self-time. Look for: graph breaks under
`--compile`, host-side gaps (sync/dataloader stalls), and fp32 QK-norm/RoPE cast cost. Open the Chrome
trace in Perfetto / `chrome://tracing`.

**Ablation (`ablation`)** is the "what did the engineering buy us?" grid. Each row = `samples/s`,
`dit_mfu`, `peak_mem_gb`. Note: `compile: false` is the repo default (`configs/train_world_model.yaml`)
and TF32 is never enabled in-repo — so `baseline` reflects the shipped out-of-the-box path.

**DDP (`ddp`)** runs the same per-GPU config on 1/2/4 H100. Weak-scaling efficiency =
`samples_per_sec_global(N) / (N × samples_per_sec_global(1))`; **< 90%** points at comms / the
per-step `dist.barrier()`.

## Results so far (H100 SXM, batch 1)

### Inference — eager vs torch.compile
| diffusion steps | eager ms/latent-frame | compile ms/latent-frame | speedup |
|---|---|---|---|
| 1 | 41.0 (2.44× rt) | 22.9 (4.37× rt) | 1.79× |
| 2 | 55.6 (1.80× rt) | 28.1 (3.56× rt) | 1.98× |
| 4 | 85.1 (1.17× rt) | 38.7 (2.58× rt) | 2.20× |
| 8 | 143.9 (0.69× rt) | 60.5 (1.65× rt) | 2.38× |
| decode | 255 ms / 274 video fps | 82 ms / 852 video fps | 3.11× |
| end-to-end @1 step | 907 ms / 1.76× rt | 440 ms / 3.64× rt | 2.06× |

Realtime bar = 10 latent fps (20 fps ÷ td 2). Marginal cost ≈ 14.6 ms/step eager, 5.3 ms/step
compiled. **Verdict: inference is well-optimized for interactive serving; `torch.compile` (shipped
OFF in `configs/eval_world_model.yaml`) ≈ doubles world-model headroom and triples decode.** Steps↔
quality is the remaining tradeoff (needs a real checkpoint's FDD curve).

#### Per-frame denoise profile (eager, `profile --mode infer`)
~29.7 ms/frame of GPU kernel time vs ~70–85 ms wall → **only ~35% GPU utilization: the per-frame
denoise is launch/overhead-bound, not compute-bound** (batch-1 interactive serving, ~10k tiny
2–3 µs kernel launches/frame). Kernel-time breakdown:
- **~45% un-fused fp32 elementwise** — QK-norm (mean/sub/pow/rsqrt/mul) + RoPE + AdaLN. This is what
  `torch.compile` fuses → the ~2× it buys. Mechanism behind fix #1.
- **~33% memory copies** — dominated by the KV-cache `torch.cat(...).clone()` per layer per step
  (`latent_world_model.py:352-365`). Structural target: a **pre-allocated ring-buffer KV-cache** would
  remove the per-step concat+clone (a real code change, not a config flip).
- **~25% matmuls** (the useful projection compute) · **~2.8% attention** (SDPA) — negligible, so **no
  custom attention kernel is warranted**; the GQA SDPA is fine.
- TF32 helps inference little (matmul+attn are the minority); its value is training-side.

#### Adopted: A3 precomputed RoPE (now the default)
Following the profile, the A1–A5 cast-reduction options were built, secured, and swept (see
`docs/optimization_plan.md`). Winner: **A3 (precomputed RoPE tables) — a bit-exact −22% per-frame
denoise win** (50.5→39.3 ms at 4 steps, ×realtime 1.98→2.54). It is now **on by default**
(`LatentWorldModelConfig.rope_precompute = True`). A1/A2 add nothing on top of it and A5 is a dead end,
so they stay off. Shipped inference stack = `compile: true` + `rope_precompute: true`, all
precision-preserving, no custom kernels.

### Training — DiT MFU across the levers
| config | DiT MFU | samples/s | peak mem |
|---|---|---|---|
| batch 1, eager (**shipped default**) | 15.7% | 7.6 | 34.3 GB |
| batch 4, eager | 24.5% | 11.84 | 61.9 GB |
| batch 8, eager | OOM | — | >80 GB |
| **batch 4, +compile** | **44.0%** | 21.3 | 44.5 GB |

(MFU ≈ samples/s × 0.0207; per-sample FLOPs is constant, so this holds for any batch/compile combo —
the JSON omits `dit_mfu` under `--compile`, compute it from samples/s.)

**Verdict: training ships under-optimized (15.7% MFU at the default batch size of 1, 34/80 GB used) —
the GPU is starved, not kernel-bound — but two config flips fix it. Batch 4 + `torch.compile` reaches
44% MFU (2.8× the default throughput) and is healthy; compile also *cuts* peak memory 28% (61.9 → 44.5
GB), so batch 8 + compile should fit and push MFU higher still.** None of this needs new kernels — it's
batch size + `torch.compile` + TF32, all one-line config changes.

Still optional (won't change the verdict): `--batch 8 --compile [--activation-ckpt]` to find the MFU
ceiling, `--tf32` (Inductor warns it's off), `ddp` for scaling, `train --mode full` for the codec share.

## Bottom line + ranked fixes

**Inference is well-optimized; training ships under-optimized but is one-config-flip away from healthy.
Neither needs new/custom kernels — the wins are `torch.compile` + batch size + TF32.**

- **Inference:** realtime for interactive serving at 1–4 diffusion steps; decode is 27–43× realtime
  (never the bottleneck); `torch.compile` ≈ 2× the world model and 3× decode. Only gap: it's OFF by
  default.
- **Training:** 15.7% MFU out of the box (batch 1) → **44% MFU** with batch 4 + compile (2.8×), and
  compile cuts memory so batch can go higher. Starved GPU, not bad kernels.

Ranked fixes (all one-line config changes):
1. **Turn `torch.compile` on by default** — `run.compile` in `train_world_model.yaml`, `compile` in
   `eval_world_model.yaml`. ~1.8× training throughput, ~2× inference, and −28% training memory. Biggest
   single win; the reproducibility argument for keeping it off is weak once it's this much faster.
2. **Raise the default training batch size** from 1 — the default uses 34/80 GB and is the root cause of
   the 15.7% floor. With compile's memory savings, batch 8 fits.
3. **Enable TF32** — `torch.set_float32_matmul_precision('high')` (Inductor literally warns it's off).
   Free win on the fp32 QK-norm/RoPE matmuls.
4. (**As batch grows**) **activation-checkpointing** — the config flag exists (`activation_checkpointing`,
   default off) to trade compute for memory and unlock even larger batches / MFU.
5. (**Inference, structural — not a config flip**) **ring-buffer KV-cache.** The per-frame denoise
   profile is launch/memory-bound, and ~33% of its kernel time is the KV-cache `torch.cat(...).clone()`
   per layer per step (`latent_world_model.py:352-365`). A pre-allocated ring buffer written in place
   would remove the per-step concat + clone. Only worth it after compile (which fuses the ~45%
   elementwise churn first). Do **not** add a custom attention kernel — SDPA is only ~2.8%.

## Caveats (no silent caps)

- **Random weights** → quality metrics (FDD/DINO-drift) are meaningless here; deliberately omitted.
  Use `eval_world_model_offline.py` + a real checkpoint for the speed↔quality tradeoff.
- **MFU numerator** = 3 × measured forward FLOPs (backward ≈ 2× forward). FLOPs are measured with
  `torch.utils.flop_counter` (handles factored attention); it does **not** compose with a compiled
  graph, so compiled runs report timing/throughput but skip the MFU field — read MFU from the
  eager run at the same batch.
- **`--peak-flops`** defaults to H100 SXM bf16 (989.5 TFLOP/s). Override for a different card.
- The **dataloader-bound test needs real data** on the volume; random-init runs skip it and say so.
- **`decode` fps counts all decoded latents** (context + generated) = pure decoder throughput; the
  **`end_to_end` video_fps counts only newly-generated frames** but its timed region re-decodes the
  context each call (a streaming server would cache it), so e2e is a deliberately pessimistic bound —
  the two rows measure different things, don't subtract them.
- **DINO encode is unchunked** here (unlike the trainer's `dino_max_chunk_size`); at large
  `--batch`/`--n-frames` the single DINOv3-L forward can spike peak memory. Fine at the defaults.
```
