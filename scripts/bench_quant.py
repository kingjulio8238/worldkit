"""Category C (matmul/quantization): weight-only int8 / fp8 on the diffusion transformer.

At batch 1 the DiT projections are skinny, weight-**bandwidth**-bound GEMMs, so shrinking the weights
(int8 = half the bytes vs bf16; fp8 similar) speeds the per-frame denoise even though FLOPs are tiny.
Applied to the trainable transformer only (the frozen codec decode is a separate, batched path).

This is a precision change: speed is measured on random init here, but *adoption* must pass the FDD
quality gate (`qualitycheck_optims.py`) on a real checkpoint -- same rule as A2/A5.

Uses torchao's `quantize_`; apply BEFORE `torch.compile` (the standard torchao+compile flow).
"""

from __future__ import annotations

VALID = {"none", "int8", "fp8"}


def apply_quantization(model, mode: str) -> str:
    """Quantize the DiT's Linear weights in place. Returns the mode applied ('none' if a no-op).

    - int8: `int8_weight_only` — halves weight bytes; best batch-1 (bandwidth-bound) lever.
    - fp8:  `float8_weight_only` — H100 e4m3 weights (needs sm89+); more useful once batched.
    """
    if mode == "none":
        return "none"
    if mode not in VALID:
        raise ValueError(f"unknown --quantize {mode!r}; valid: {sorted(VALID)}")

    from torchao import quantization as q  # noqa: PLC0415

    # torchao renamed the function API (int8_weight_only) to config classes (Int8WeightOnlyConfig);
    # support both so the bench works across torchao versions.
    if mode == "int8":
        cfg = q.Int8WeightOnlyConfig() if hasattr(q, "Int8WeightOnlyConfig") else q.int8_weight_only()
    else:  # fp8
        cfg = q.Float8WeightOnlyConfig() if hasattr(q, "Float8WeightOnlyConfig") else q.float8_weight_only()

    inner = getattr(model, "single_world_model", model)  # unwrap the multiplayer wrapper
    q.quantize_(inner.world_model, cfg)
    return mode
