# Inference denoise optimization plan

Goal: minimize per-frame streaming-denoise latency (the interactive serving cost). Grounded in the
`profile --mode infer` kernel breakdown (batch 1, 4 diffusion steps, H100).

## Current state (per-frame denoise)

| | kernel time | wall/frame (4 steps) | GPU util |
|---|---|---|---|
| eager | 29.7 ms | ~85 ms | ~35% |
| compiled | 22.5 ms | 38.7 ms | ~58% |

**Compiled kernel composition:** dtype casts (`_to_copy`) **~50%**, matmul ~28%, attention ~5%,
norm-math ~2%, rest memory/misc. compile fused the elementwise *math* (was ~45% eager → ~2%); what
remains is **bf16↔fp32 casts (QK-norm/RoPE)** + **launch gaps** + the matmul floor.

**Theoretical floor:** if casts + memory copies + launch gaps are removed, we approach the matmul floor
(~6 ms/frame kernel) with util → ~90% ⇒ plausibly **~10–15 ms/frame wall**, i.e. ~2.5–3× beyond compile.
Numbers below are estimates to be confirmed by re-profiling after each step.

Note: matmuls run in **bf16** (tensor cores) under autocast → **TF32 is ~irrelevant for inference**
(it only affects fp32 matmuls). TF32 is a training-side knob at best.

---

## A. Dtype casts + elementwise (QK-norm / RoPE / AdaLN) — ~50% post-compile — HIGHEST
Root: `ml/attention.py` QKLayerNorm/QKRMSNorm do `x.to(float32) → math → .to(bf16)`; `apply_rotary_emb`
does `x.float()*cos + …`. Each forces a bf16→fp32→bf16 round-trip (2× the bytes moved).

- **A1. Fused RMSNorm** — replace `QKRMSNorm` with `F.rms_norm` (torch ≥2.4) or an Apex/Triton fused
  RMSNorm that upcasts *inside* the kernel (no materialized fp32 tensor). Removes a large cast chunk.
  Effort: low. Risk: low (numerically equivalent).
- **A2. Fused RoPE** — Triton/flash-attn RoPE kernel with bf16 I/O + fp32 accumulation; precomputed
  cos/sin. Removes the `x.float()` materialization. Effort: med. Risk: low–med.
- **A3. Cache RoPE tables** — `self.rope(rope_len)` recomputes cos/sin every call; memoize per
  (rope_len, device). Effort: trivial. Impact: small–med.
- **A4. Mega-fuse QK-norm + RoPE + SDPA input-prep** into one Triton kernel (they're adjacent per head).
  Effort: high. Impact: high (kills the remaining cast/elementwise traffic in one pass).
- **A5. Lower precision for norm/RoPE** — bf16 QK-norm with eps tune, or fp16 RoPE tables. Effort: low.
  Risk: HIGH — gate on the FDD quality curve before keeping.
- **A6. Inductor tuning** — `mode="max-autotune"`. **RESULT: out.** (1) It **enables CUDA-graph trees**,
  so it errors with the same KV-cache-as-graph-output conflict as E1's `reduce-overhead`. (2) The
  autotune log (before the crash) shows the **default cuBLAS `mm`/`bias_addmm` wins at 100%** for every
  large GEMM (e.g. cuBLAS 0.086 ms vs best Triton 0.109 ms) → **zero matmul headroom** even without the
  crash. `ALLOW_TF32=False` throughout also re-confirms the bf16 path (TF32 inert). Effort: trivial;
  payoff: none.

## B. Memory copies / KV-cache — ~33% eager — HIGH (structural)
Root: `world_model/latent_world_model.py:352-365` rebuilds each layer's cache with
`torch.cat([k_ctx[:, :nreg], k_ctx[:, nreg+1:], new_k]).clone()` every step; plus defensive `.clone()`s.

- **B1. Ring-buffer static KV-cache** — pre-allocate `(B, ctx_window, n_kv_heads, head_dim)` per layer,
  write new K/V in place at a rolling index, drop the cat+clone. Effort: med. Impact: high.
  **Also the enabler for CUDA graphs (E1).**
- **B2. Guard the register-token slice/cat** — `n_register_tokens == 0` by default, so
  `k_ctx[:, :0], k_ctx[:, 1:]` is a no-op that still allocates a new tensor. Skip when 0. Effort: trivial.
- **B3. Drop the defensive `.clone()`s** once a static buffer removes the compile-aliasing concern.
- **B4. Contiguous append via index** instead of re-`cat`.
- **B5. Paged/streaming attention** that writes KV in-kernel (larger change; pairs with A4).

### B1 status — IMPLEMENTED (opt-in, bit-exact) — the E1 enabler
`WorldModelInferenceConfig.streaming_cache: "grow" | "ring"` (default `"grow"`).
`"ring"` updates the fixed prefill KV buffer **in place** (`torch.roll` the context region + write the
new frame at the end), keeping logical order and a stable buffer identity/shape across frames, instead
of `cat`+`clone`-ing a fresh tensor (`latent_world_model.py denoise_streaming`). Wired through
`rollout.py` (`ring_cache=config.streaming_cache=="ring"`).

**Verified bit-exact:** full random-init rollout, grow vs ring, `noise_level=0` → **maxdiff = 0.00e+00**
(logical order preserved ⇒ attention sees identical keys in identical order ⇒ no reduction reordering).
Unit tests in `tests/world_model/test_ring_cache.py` (roll-update == cat-update, buffer identity kept).

**A/B result (compile + A3-default, median-of-5): ring REGRESSES, growing with step count.**
| n_steps | grow | ring | Δ |
|---|---|---|---|
| 1 | 21.6 | 23.4 | +8% |
| 2 | 26.6 | 30.4 | +14% |
| 4 | 36.5 | 43.7 | +20% |
| 8 | 56.6 | 68.9 | +22% |

The regression scales with *steps* (marginal cost 5.0→6.3 ms/step), not frames — so it is **not** the
once-per-frame roll+copy update. Root cause: reusing the same KV buffer object as an input to the
compiled `world_model` every step triggers `torch.compile`'s input-mutation handling (defensive
clone/guards per call); `grow` hands a fresh tensor each frame and avoids it.

**Verdict: keep `streaming_cache: "grow"` default; do NOT adopt ring standalone.** B1 is bit-exact and
correct but regresses under plain compile.

**Diagnostic (profile grow vs ring, compile, 4 steps) — the mechanism, corrected:**
| | Self CUDA | ProfilerStep wall | added kernels |
|---|---|---|---|
| grow | 66.37 ms | 150.4 ms | — |
| ring | 66.44 ms | **128.6 ms** | one `aten::copy_` (0.94%) |

GPU kernel work is **identical** (the `_to_copy` counts match byte-for-byte); ring adds only the tiny
roll/copy update. The earlier hypothesis (per-step input-clone *kernels*) was **wrong**. The +20%
benchmark regression is **CPU-side**: the in-place-mutated buffer bumps its version counter, so dynamo
re-checks guards / recompiles the compiled `world_model` **per diffusion-step call** — which is why the
regression scaled with steps. The profiler (persistent buffer, prefill excluded) does not trip this and
even shows ring *wall-faster* (128.6 vs 150.4 ms) — a preview of graph-captured behavior.

**→ E1 (CUDA graphs) is the right lever, but the cheap path is closed.** Captured replay bypasses the
dynamo guards that cause the regression — but the cheap test (`torch.compile(mode="reduce-overhead")`,
auto CUDA-graph trees) **errors**:

> `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`

The streaming **KV-cache is a graph *output*** (prefill `return_kv`) read back by later graph runs;
cudagraph-trees pools that output and the next replay clobbers it — before the next diffusion step reads
it. Hits **both grow and ring** (grow's `.clone()` is post-loop, not on the prefill output). The
suggested `cudagraph_mark_step_begin()` silences the error but corrupts results (we genuinely reuse the
cache across runs). **So E1 requires the real work: the KV-cache must be a *static* tensor the compiled
region writes in place (not returns), living outside the cudagraph pool** — manual static-buffer
management or a hand-captured `torch.cuda.graph` of the per-step denoise. Large, GPU-only-testable,
bounded upside (the ~42% launch-idle). Decision point: invest in full manual E1, or bank A3 (shipped,
bit-exact −22%) and defer E1. B1's ring buffer is the correct substrate for the static-buffer version.

## C. Matmuls (projections) — ~28% — the useful floor; cut via bandwidth
At batch 1, M≈144 tokens ⇒ skinny GEMMs that are **weight-bandwidth bound**, not FLOP bound.

- **C1. Weight-only int8/int4** on the Linear layers → less weight bandwidth → faster skinny GEMM.
  Biggest matmul lever at batch 1. Effort: med. Risk: med (gate on FDD).
- **C2. fp8 (H100 e4m3)** matmuls → ~2× tensor-core throughput. Effort: med–high. Risk: med.
- **C3. Batched multi-stream serving** → fat GEMMs, tensor-core saturation. Helps *throughput*, not
  single-stream latency. Effort: med.
- **C4. Epilogue fusion** (bias/residual into the GEMM) — compile does some; verify.
- **C5. TF32 — SKIP for inference** (matmuls are bf16; TF32 only helps fp32 matmuls).

### C status — IMPLEMENTED (bench path, measurement pending)
`bench_quant.apply_quantization(model, {int8,fp8})` — torchao weight-only quant on the DiT's Linears
(132 of them), applied before compile. Flags: `bench_infer_speed --quantize {none,int8,fp8}` + Modal
`infer quantize=`. torchao pinned to 0.12.0 in the image (torch-2.8 ABI; the latest targets 2.11 and
skips its cpp extensions → slow int8 fallback). Validated on the real DiT (CPU): int8 quantizes 132
Linears, forward runs, loss rel-diff 0.6% (expected — **precision change, FDD-gate before adoption**).
C3 (batched) = `--batch >1`; A6 (max-autotune) = `--compile-mode max-autotune`; C4 (epilogue) is handled
by compile (observe, no code).

**C1 int8 result (batch 1, compile + A3): REGRESSES, growing with steps.**
| n_steps | baseline | int8 | Δ |
|---|---|---|---|
| 1 | 21.6 | 24.5 | +13% |
| 4 | 36.5 | 45.2 | +24% |
| 8 | 56.6 | 74.1 | +31% |

**Unifying insight — the per-frame denoise is launch-bound (~58% util), so only launch-*count*
reductions help; bandwidth/compute savings are masked by the gaps.** This explains the whole arc: A3
won by *removing* RoPE-recompute launches; A1/A2 cut kernel time (not count) → no help; int8 *adds* a
dequant launch per matmul → +24%; E1 (graphs, eliminate launches) is the real remaining lever.
**Consequence: quantization (C1/C2) is a batched-throughput lever, not a single-stream-latency one** —
re-test at batch ≥8 where GEMMs are large and launch overhead is amortized. Don't adopt int8/fp8 for
single-stream serving.

**C1 int8 @ batch 8 — CONFIRMED (the sign flips):**
| n_steps | batch-8 baseline | int8 | Δ |
|---|---|---|---|
| 1 | 97.9 | 95.9 | −2% |
| 4 | 130.1 | 120.5 | −7% |
| 8 | 173.0 | 153.6 | −11% |

int8 flips from +24% (batch 1) to −7…−11% (batch 8), win growing with steps (more matmul). Confirms the
launch-bound→compute-bound crossover.

### C verdict — FINAL
- **Interactive single-stream (batch 1, the model's real use case): do NOT quantize** — int8 +24% (launch-bound).
- **Batched / offline generation (batch ≥8): int8 is a valid −7…−11% throughput lever**, growing with
  step count. Opt-in via `apply_quantization(model, "int8")` (bench flag `--quantize int8`) at serve
  time — **precision change, FDD-gate before shipping** (int8 loss drift ~0.6% on random init). Not
  added to the model config (it's a serving-time weight transform, not architecture).
- **fp8 (weight-only) is OUT** — measured +27…+60% at batch 8. `Float8WeightOnlyConfig` keeps activations
  in bf16, so there is **no fp8 tensor-core matmul path** (that needs *both* operands fp8); it just
  dequants fp8→bf16 each forward = pure overhead. Real fp8 speed needs
  `Float8DynamicActivationFloat8WeightConfig` (activations fp8 too) — a larger precision change (higher
  FDD risk) not worth chasing since int8 already gives the batched win. A6 optional (launch-bound caps it).

## Category sweep — non-kernel items (E2/E3/E4): scoped, not benchable as flags
These are the only untouched plan items; they are **algorithmic / serving-architecture**, not
kernel-level flags, so they are characterized here rather than measured by a bench run:
- **E2. Fewer diffusion steps / step-distillation** — the single biggest wall-clock lever (latency is
  ~linear in steps: with graphs 1-step 20.3 ms vs 4-step 33.4 ms ⇒ ~1.6×). The speed is already
  measured; the open question is **quality**. Harness: `qualitycheck_steps.py` (+ Modal
  `qualitycheck_steps`) runs the FDD/DINO-drift metrics at each step count on a real checkpoint and
  reports the **min viable steps** (fewest within 5% of the max-step quality). **Now runnable** — the
  weights are public (below); no training needed to *characterize* it, only to push further.

  **Available checkpoints (public, ungated, HF, as of 2026-07-20 — the "unlock at launch" gate is stale):**
  Alakazam's MIRA-Mini reproductions (the 5B MIRA itself ships code+dataset only, no weights):
  - `alakazamworld/mira-mini` — 1B single-player (`checkpoint-52000`, ~20 GB) — the base for the sweep.
  - `alakazamworld/mira-mini-psd` — **1B, already a 2-step progressive-self-distillation variant** — the
    distillation E2 would produce, ready to compare against the base at low steps.
  - `alakazamworld/mira-mini-4p` (multiplayer), `alakazamworld/mira-mini-364m` (distilled student).
  Layout matches our loader exactly (`world_model_config.yaml` + `checkpoint-*/checkpoint.pth` + `codec/`
  + `context/`). **Dataset `kyutai/rocket-science` is gated** (HF login + accept CC-BY-NC-SA terms); the
  `test/` split has the `index.json` the eval needs. CC BY-NC-SA 4.0, non-commercial.

  **The MIRA paper (arXiv 2607.05352, §6.4 / Fig 11) already answers the core E2 question:** it sweeps
  flow-matching steps {1,2,4,6,8,10} for the baseline vs a **PSD self-distilled** model. Inference
  default = **10 steps** (linear-quadratic schedule, context re-noise 0.2). Finding: the **undistilled
  baseline degrades sharply below ~4-6 steps; the PSD-distilled model stays stable down to 1-2 steps**
  (PSD wins at every step count, by a wide margin in the few-step regime). Baseline latent-WM quality:
  gFDD 0.55 / gFID 10.7 / gFVD 163.1.

  **⇒ E2 conclusion (research-backed):** you can't just lower `n_diffusion_steps` on the base model —
  quality collapses. The win is the **distilled model at few steps**, and it already exists as
  `mira-mini-psd` (2-step). So the E2 lever = **run the 2-step PSD model instead of the 10-step base**
  ≈ **~5× fewer denoise forwards at maintained quality** — and our compile+A3+E1 stack applies on top of
  it (2-step + graphs = 24.7 ms/frame in our bench). Our `qualitycheck_steps` sweep VERIFIES this on the
  real checkpoints (base degrades, psd holds).

  **Gotcha:** the FDD metric needs Meta-gated **DINOv3** weights (`torch.hub` + `RS_DINO_WEIGHTS_DIR`),
  separate from the codec's DINO. So a full FDD run needs: mira-mini weights (public), the rocket-science
  test split (HF-gated), and DINOv3 weights (Meta-gated). The step-count SPEED is already measured and
  weight-agnostic; only the quality curve needs these.
- **E3. Overlap decode with next-frame denoise** — pipeline the frozen-codec decode (already 27–43×
  realtime, non-bottleneck) against the next frame's denoise to hide it in end-to-end serving. A serving
  -loop change, not a kernel change; only matters for the decode-inclusive path.
- **E4. Whole-step mega-kernel / captured graph** — the fusion limit of A4 + B1 + E1 (one captured graph
  per denoise step). Superset of the deferred E1; same static-buffer prerequisite and GPU-only-testable
  risk. Deferred with E1.

## D. Attention — 2.8% — LEAVE ALONE
Already FlashAttention via SDPA. No custom attention kernel is warranted. Revisit only if the context
window grows a lot.

## E. Cross-cutting — attack launch overhead + step count — HIGH
- **E1. CUDA graphs** — `torch.compile(mode="reduce-overhead")` or manual capture eliminates per-kernel
  launch gaps (the 35→58% util story; graphs push toward ~90%). Requires static shapes ⇒ needs **B1**.
  Impact: HIGH in this launch-bound regime.

### E1 status — IMPLEMENTED (manual whole-frame capture, opt-in) — awaiting GPU measurement
Manual `torch.cuda.graph` capture of the **whole per-frame denoise** (all diffusion steps + kv-update +
ring rotation), replayed per frame. Config: `WorldModelInferenceConfig.cuda_graphs: bool = False`
(forces the ring cache; single-player, `n_register_tokens==0`, PSD off — guarded). Bench/Modal flag
`--cuda-graphs` / `cuda_graphs=True`.
- **`FrameGraphRunner`** (`inference/cuda_graphs.py`): owns static I/O buffers + the ring KV buffers;
  lazily captures on the first frame (warmup on a side stream, save/restore the real prefill so warmup
  doesn't corrupt it), replays per frame. **Any capture/replay failure disables graphs and falls back
  to eager** — a failed capture never crashes the run.
- **Refactor:** `denoise_streaming` split into `_prefill_cache` (eager, dynamic) + `_denoise_frame_body`
  (fully static: baked schedule, in-place ops, no data-dependent control flow) — the capture unit.
  Behaviour-preserving (21 rollout/inference tests pass).
- **Validated (CPU):** guard logic + eager-fallback bit-exactness (`cuda_graphs` on CPU → capture fails
  → eager ring == grow). Guard unit tests in `tests/inference/test_cuda_graphs.py`.
- **GPU RESULT: captured cleanly (no fallback), and it works.** vs shipped baseline (compile+A3, grow):
  | n_steps | baseline | +graphs | Δ | graphs on eager |
  |---|---|---|---|---|
  | 1 | 21.6 | 20.3 | −6% | 22.7 |
  | 4 | 36.5 | 33.4 | −8.5% | 39.3 |
  | 8 | 56.6 | 51.5 | −9% | 61.5 |

  **compile + graphs is the new best (−8.5% @ 4-step, ×realtime 2.74→2.99), win growing with steps, and
  it ERASES B1's +20% regression** (the ring is a win once graphs own the buffer). Notably, **graphs on
  *eager* kernels nearly match compiled** (39.3 vs 36.5) — graphs alone recover most of compile's benefit
  by killing launch overhead, confirming the launch-bound diagnosis. Realized ~1.1× (below the ~1.4×
  estimate — un-graphed prefill/decode + input-copy overhead + compute floor cap it), but a clean win on
  the first GPU try.
- **CORRECTNESS: BIT-EXACT.** `infer --verify-graphs` (graphed vs eager rollout, noise_level=0, fixed
  seed) → **maxdiff = 0.0**. The graph path is provably correct, not just fast.
- **VALIDATED & ADOPTABLE.** Opt-in `cuda_graphs: true` for single-stream serving, stacked on
  compile+A3 → **33.4 ms/frame @ 4-step, ×realtime 2.99** (the fastest measured). Kept off by default
  (single-player / nreg==0 / PSD-off constraints + one-rollout-at-a-time shared buffers; auto-falls back
  otherwise). Full single-stream stack: eager ~50 → A3 39.3 → +compile 36.5 → +graphs **33.4** ms/frame.
- **E2. Fewer diffusion steps (algorithmic)** — 1–2 steps are already realtime; distill to 1-step. The
  single biggest wall-clock lever, gated by quality.
- **E3. Overlap decode with next-frame denoise** (pipeline) for end-to-end serving.
- **E4. Whole-step mega-kernel / captured graph** — the limit of A4+B1+E1.

---

## Recommended sequence (max impact / least risk first)
1. **B1 static ring-buffer KV-cache + E1 CUDA graphs** — removes launch gaps *and* the ~33% copy cost;
   targets the 42% idle. Biggest single jump. (B2/B3 fall out of this.)
2. **A1 fused RMSNorm + A2 fused RoPE (+ A3 cached tables)** — removes ~half the remaining kernel time
   (the ~50% casts).
3. **C1 weight-only int8** — if still matmul/bandwidth-bound at batch 1.
4. **A4 mega-fuse** and **E2 step distillation** — the aggressive frontier.
5. **Quality gate (FDD via `eval_world_model_offline.py` on a real checkpoint) after ANY precision or
   step change** (A5, C1, C2, E2).

Estimated trajectory (confirm by re-profiling): compiled ~38.7 ms/frame → +B1/E1 ~25 ms → +A1/A2 ~15 ms
→ +C1 ~10–12 ms. ≈3× beyond compile ⇒ ~10× realtime at 4 steps, or 1-step realtime with large margin.

## Measured results — A1–A5 sweep (H100, batch 1, compile ON, ms/latent-frame)

| variant | 2-step | 4-step | Δ (4-step) | verdict |
|---|---|---|---|---|
| baseline (compile only) | 34.4 | 48.9 | — | reference |
| A1 fused QK-norm | 31.1 | 45.0 | −8% | clean, zero precision change |
| A2 bf16 RoPE | 32.1 | 44.4 | −9% | clean |
| A3 cached RoPE tables | 30.3 | 42.7 | −13% | biggest single win; recompiles under compile (see below) |
| **A4 = A1+A2+A3** | 32.3 | **41.8** | **−15%** | **best** |
| A5 full bf16 | 32.2 | 47.2 | −3% | slower than A1/A2 + quality risk → skip |

(1-step numbers for A3/A4 showed a ~5200 ms first-frame **compile artifact** — the warmup only warms
the 4-step graph and A3's Python-dict cache forces a recompile at the new step count — so compare on
the warmed 2/4-step columns.)

**Verdict:** A4 is ~15% faster than compile-alone at 4 steps (48.9→41.8 ms/frame, ×realtime 2.04→2.39),
on top of compile's ~2×. **A3 (cached RoPE) carries most of the win and is precision-free**, but its
Python-dict cache **doesn't compose with `torch.compile`** (recompiles on shape change) — bank it as a
**compile-native precomputed RoPE buffer**, not a runtime dict. A1+A2 are compile-clean. **A5 (full
bf16) is a dead end** — slower than A1/A2 (manual bf16 fuses worse than `F.rms_norm`) and quality-risky.
The cast optims are a modest squeeze vs compile/batch; the larger remaining inference win is structural
(B1 ring-buffer KV-cache + E1 CUDA graphs against the ~33% memory + launch overhead).

## Implementation status — SECURED (real config-flagged code + tests + quality gate)

A1–A3 are now **real, opt-in model code** behind config flags (defaults preserve released behaviour);
A5 stays bench-only (dead end). The six "secure A" steps are done:

| step | what | where |
|---|---|---|
| 1 | benchmark artifacts fixed — warm each `n_diffusion_steps` graph separately + median-of-N (kills the ~5200 ms A3/A4 spike, de-noises the A1↔A4 ranking) | `bench_infer_speed.py` (`--repeats`) |
| 3 | **A3 compile-native** — precomputed RoPE cos/sin **buffers** (sliced per call), not a runtime dict → no torch.compile recompile | `world_model/layers/rope.py` (`enable_precompute`), flag `rope_precompute` |
| 4 | **A1/A2 promoted to config flags** — `qk_norm_fused` (fused `F.layer_norm`/`F.rms_norm`), `rope_upcast=False` (bf16 RoPE) | `ml/attention.py`, `SelfAttentionConfig` + `LatentWorldModelConfig`, wired via `DiffusionTransformer` |
| 4c | **unit tests** — A1 fused≈eager (tight tol), A2 bf16 bound, A3 precompute==recompute exactly, config-flag plumbing | `tests/ml/test_attention_optims.py` (9 pass) |
| 2 | **quality gate** — FDD/drift per variant on a real checkpoint, regression report | `qualitycheck_optims.py` + Modal `qualitycheck` (needs gated weights on the volume) |
| 5/6 | re-benchmark via the secured flags; **A5 dropped** (slower + quality-risky) | `optim_sweep`, `--optim` |

Flags are read at **forward** time, so `bench_optims.apply_optims(model, flags)` toggles them on a
built model with no rebuild (used by the benches and the quality gate); `config_overrides_for(flags)`
gives the equivalent construction-time overrides for the production path. Validated: real-flag outputs
match the model bit-for-bit for A3, within 1e-5 for A1, and are a bounded bf16 approximation for A2.

**Re-benchmark (secured, artifact-free):**
`modal run scripts/modal_bench.py::optim_sweep` — now with warm-per-step + median, A3 compile-clean.
**Quality-gate (needs a checkpoint on the volume):**
`modal run scripts/modal_bench.py::qualitycheck --checkpoint /data/<ckpt>.pth`.

### Secured-harness results (compile ON, median-of-5, artifact-free) — FINAL
| variant | 1-step | 2-step | 4-step ms | Δ (4-step) | ×realtime (4-step) | precision |
|---|---|---|---|---|---|---|
| baseline | 27.0 | 35.3 | 50.5 | — | 1.98 | exact |
| A1 fused QK-norm | 27.7 | 36.4 | 52.8 | +5% | 1.89 | exact |
| A2 bf16 RoPE | 24.8 | 31.7 | 44.7 | −11% | 2.24 | bf16 |
| **A3 precomputed RoPE** | 22.6 | 28.3 | **39.3** | **−22%** | **2.54** | **exact** |
| A4 = A1+A2+A3 | 22.2 | 27.8 | 39.2 | −22% | 2.55 | bf16 |
| A5 full bf16 | 26.2 | 34.4 | 49.5 | −2% | 2.02 | bf16 |

**The de-noised, compile-native harness gave a decisive and surprising result:**
- **A3 (precomputed RoPE tables) is the clear winner: −22% at 4 steps (50.5→39.3 ms; ×realtime 1.98→
  2.54), and it is bit-exact (no precision change).** Securing it (compile-native buffers, no recompile)
  *increased* its measured win from the earlier noisy −13% to −22%.
- **A4 ≈ A3 (39.2 vs 39.3): A1 and A2 add nothing on top of A3.** A2's standalone −11% does not stack —
  once the RoPE tables are precomputed, the bf16-RoPE apply saving is in the noise.
- **A1 is noise; A5 is ~baseline and quality-risky.**

### Adopt / drop decision — FINAL
**Adopt A3 only: `rope_precompute: true`.** −22% at 4 steps, ×realtime 1.98→2.54, **bit-exact**.
- **A1 → skip** (no measured benefit).
- **A2 → skip** (redundant given A3; would add a bf16 precision change for no marginal speed).
- **A5 → drop** (slower than A3 + quality-risky).

**Consequence: the FDD quality gate is not needed for the shipped config** — the only adopted optim (A3)
produces identical outputs (unit-tested exact). The quality harness stays for future precision-changing
work (e.g. quantization in tier C). Net inference stack: `compile: true` + `rope_precompute: true` →
~2× (compile) × 1.28 (A3) headroom, all precision-preserving, no custom kernels.

## Non-goals (the profile rules these out)
- **No custom attention kernel** (SDPA is ~3–5%).
- **No TF32 for inference** (bf16 matmuls).
- Don't hand-fuse the elementwise *math* — `torch.compile` already did (45% → 2%); the residue is casts,
  which need fused norm/RoPE, not more elementwise work.

---

# E2 execution plan — adopt the 2-step PSD checkpoint (path to ~7×)

The full-stack single-stream speedup so far is **~2.6×** vs the released eager baseline (compile + A3 +
E1 CUDA graphs, all bit-exact). The remaining large lever is **algorithmic**: run **fewer diffusion
steps**. The released default is **10 steps**; the base model degrades sharply below ~4–6 steps
(arXiv 2607.05352 Fig 11), but the **`mira-mini-psd`** checkpoint (progressive self-distillation) holds
quality down to **1–2 steps**. Stacking the two independent multipliers:

| lever | needs | factor |
|---|---|---|
| step distillation 10 → 2 | `alakazamworld/mira-mini-psd` (public) | ~3.1× (173 → 55.6 ms eager) |
| our engineering stack (compile + A3 + E1 graphs) | shipped | ~2.3× (55.6 → 24.1 ms) |
| **combined** | both | **~7.2×** (173 ms 10-step-eager base → **24.1 ms** 2-step full-stack) |

Baseline = released MIRA-**Mini** (1B, 10 steps, eager, compile off) ≈ 173 ms/frame (extrapolated from
the measured 1/2/4/8-step curve). Target = 2-step PSD + full stack = **24.1 ms/frame (~41 fps, 4.1×
realtime)**. All on **MIRA-Mini 1B**, not the 5B MIRA (unreleased). The *speed* is already measured
(weight-agnostic); the gate is *quality* — proving 2-step PSD holds.

## Phase 0 — Graph tier decision → **Tier B (chosen)**
Because a PSD checkpoint feeds an extra per-step input (`tau_delta`, the integration step size,
`latent_world_model.py:345`) to the denoiser, E1 CUDA graphs were guarded off for PSD. **Tier B** makes
graphs work with PSD so the 2-step path reaches the full 24.1 ms (Tier A = compile+A3 only, ~27 ms).
Chosen: **Tier B.** The `tau_delta` is derived from the already-static `delta_ts` inside the captured
`_denoise_frame_body`, so the fix is to drop the conservative PSD guard (no new static buffer needed).
- **Exit:** guard removed; `--psd` bench flag exercises the PSD inference path on random init.

## Phase 1 — Stage assets on the volume  *(no paid GPU)* — ✅ DONE (2026-07-20)
`modal run scripts/modal_bench.py::stage_assets` downloaded onto the `mira-bench-data` volume; verified
by `::verify_assets`:
- base: `/data/checkpoints/mira-mini/checkpoint-52000/checkpoint.pth` (+ bundled `codec/checkpoint-125000/`).
- PSD: `/data/checkpoints/mira-mini-psd/checkpoint-10000/checkpoint.pth` (+ bundled codec).
- dataset: `/data/datasets/rocket-science/test` — `index.json` OK, **62 samples**.

Two follow-ups this surfaced:
- **DINOv3 metric weights (Phase 2 prerequisite).** The FDD *metric* builds its own DINOv3 backbone
  (separate from the codec's, which is restored from the codec checkpoint). It reads
  `RS_DINO_WEIGHTS_DIR`; with nothing there it **silently uses a random-init DINO**, making
  `dino_frechet` / `dino_*_drift` meaningless. `stage_dino` stages the Meta-gated `dinov3_vitb16`
  weights to `/data/dino_weights`, and both quality functions now export `RS_DINO_WEIGHTS_DIR` to it.
  **Inception-FID + latent-drift + LPIPS/PSNR/SSIM need no DINO weights and stay valid** as a fallback.
- **62-sample test split** is small for Frechet stats — treat the curve as *relative* (base-vs-PSD,
  steps-vs-steps), and cap `--num-samples` at ~62 so the loader isn't asked for more clips than exist.

## Phase 2 — Quality gate (the proof)  *(paid GPU, ~1–2 H100-hr)*
Optionally `modal run ...::stage_dino` first (valid DINO metrics). Then run `qualitycheck_steps` on base
and PSD over `1 2 4 8 10` steps; report min-viable steps (within 5% of 10-step quality).
- **Expect:** base min-viable ≈ 4–6; **PSD min-viable = 1–2.** That green-lights 2 steps.

## Phase 3 — PSD graph wiring + correctness  *(Tier B code, done in Phase 0; verify here)*
- `bench_infer_speed --compile --cuda-graphs --psd --verify-graphs` → **maxdiff 0.0** (PSD graphed ==
  PSD eager), the same bar A3/E1 pass. Unit test: PSD-enabled graphed rollout == eager (CPU fallback).
- **Exit:** bit-exact PSD graph path.

## Phase 4 — Confirm headline speed  *(paid GPU, minutes)*
- `modal run ...::infer --n-diffusion-steps "1 2" --compile --cuda-graphs` (+ `--psd`) →
  2-step ≈ **24 ms/frame, ~4.1× realtime** — closes the ~7× claim on the PSD path.

## Phase 5 — Make it the serving default  *(code + docs)*
- Serving config: PSD checkpoint, `n_diffusion_steps=2`, `schedule_type=linear_quadratic`,
  `noise_level=0.2`, `cuda_graphs=true`, compile on.
- Document the 10-step-eager-base → 2-step-PSD-full-stack = ~7× result; commit + push to worldkit.

## Status / critical path
| phase | status | GPU $ | blocker |
|---|---|---|---|
| 0 Tier B graphs | ✅ code done | none | — |
| 1 stage assets | ✅ staged + verified | none | — |
| 2 quality gate | ⏳ awaits run | ~1–2 hr | DINOv3 metric weights (`stage_dino`); else FID/latent-drift only |
| 3 PSD graph verify | ⏳ awaits run | minutes | none (random-init, no deps) |
| 4 speed confirm | ⏳ awaits run | minutes | none (speed already measured) |
| 5 defaults + docs | pending | none | gated on Phase 2 result |

**Progress:** Phase 0 (Tier B PSD graphs) + Phase 1 (assets staged + verified on the volume) done.
Phase 3 (`infer --compile --cuda-graphs --psd --verify-graphs`, bit-exact) and Phase 4 (2-step speed)
are runnable now with no further deps; Phase 2 additionally wants `stage_dino` for valid DINO metrics.
