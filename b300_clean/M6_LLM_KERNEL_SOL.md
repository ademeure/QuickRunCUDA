# M6 — LLM Kernel SoL Ladder (2026-04-21)

**Single-block speed-of-light for common LLM inference primitives** at 1024 elements.
Synthesizes V5 G-series + V4 Q-series kernel benchmarks.

System: B300 SXM6 AC, 1500 MHz lock, single SM, 256 threads × 4 elements/thread.

---

## SoL Ladder (single-block 1024 floats)

| Kernel | cy/op | ns @ 1500 MHz | per-element | Source |
|--------|------:|--------------:|------------:|--------|
| Argmax | **441** | **294** | 0.43 cy | G5 (#f168e50) |
| Sum reduce (Q2) | 496 | 330 | 0.48 cy | Q2 (#6b70a02) |
| RMSNorm | **588** | **392** | 0.57 cy | G4 (#470ac9c) |
| SoftMax (3-pass) | **899** | **599** | 0.88 cy | G2 (#1221b47) |
| Prefix scan (H-S) | 966 | 644 | 0.94 cy | Q4 (#5840b3d) |
| LayerNorm | **997** | **665** | 0.97 cy | G3 (#ff749ae) |
| Top-K (K=4) | **1858** | **1239** | 0.45 cy/elem/K | G6 (#ee01e84) |
| Bitonic sort | 29398 | 19600 | (n log² n) | Q5 (#a7e068c) |

---

## Key recipe: numerically-stable softmax (most critical)

```cuda
// 3-pass: max, exp+sum, normalize
1. max-reduce 1024 → 1 value (warp + cross-warp via SMEM)
2. exp(x[i] - max) via ex2.approx.ftz × 1.4427f
   sum these + reduce
3. y[i] = exp / sum (rcp.approx.ftz)
```

Per-step cost (estimated):
- Max reduce: ~150 cy
- Exp + sum reduce: ~500 cy (4 ex2 per thread + reduce)
- Normalize + write: ~250 cy

**Used by**: attention softmax, output logits, MoE gating.

---

## Key recipe: RMSNorm (transformer block norm)

```cuda
1. sumsq = sum(x[i]^2)
2. scale = rsqrt(sumsq / N + eps)
3. y[i] = x[i] * gain[i] * scale
```

Single reduction pass. Fast because no mean step (vs LayerNorm).
**Used by**: LLaMA, GPT-NeoX, Mistral, Mixtral pre-norms.

---

## Key recipe: LayerNorm (legacy)

```cuda
1. mean = sum(x) / N
2. variance = sum((x - mean)^2) / N
3. y[i] = (x[i] - mean) * rsqrt(var + eps) * gain[i] + bias[i]
```

2 reduction passes (mean, variance) + extra subtract.
**1.7× slower than RMSNorm** for same dimension.
**Used by**: BERT, original Transformer, GPT-2.

---

## Scaling guidance: 1024 → larger D

For D > 1024 elements per token:
- **D = 4096 (LLaMA-7B)**: tile across blocks; per-D RMSNorm ≈ 1.5-2 µs
- **D = 8192 (LLaMA-13B)**: ~3-4 µs RMSNorm
- **D = 14336 (LLaMA-70B FFN)**: multi-block reduction needed; ~5-7 µs

For inference throughput estimates:
- Per-token FFN+attn = ~50-100 µs at 1024-D, scales linearly
- Token throughput = batch / step_time

---

## When to multi-block

Single-block (this benchmark): up to ~1024-2048 elements per "row"
- Beyond: multi-block parallel reduction
- Tools: cooperative_groups for grid-wide reduce
- Or 2-stage: per-block → host-side gather → final

---

## Practical inference cost estimates (B300 1500 MHz lock)

Based on these single-block primitives, projected attention block:
- Q,K,V projection: cuBLAS GEMM (multi-block; not benchmarked here)
- Attention softmax (1024 head_dim): 600 ns × N_heads
- LayerNorm/RMSNorm: 400-700 ns each (input + post-attn + post-MLP = 3 norms per layer)
- Activation (GELU/SiLU): ~400 ns per FFN dim
- Output projection: cuBLAS GEMM

For **LLaMA-7B at D=4096** (32 layers, 32 heads, head_dim=128):
- Per-layer norms: 3 × 1.5 µs = 4.5 µs (RMSNorm at D=4096)
- Attention softmax: 32 heads × ~150 ns/softmax (head_dim=128) = 4.8 µs
- Total norm + softmax = ~9 µs/layer × 32 = ~288 µs/token (excluding GEMMs)

---

## SoL refinement opportunities

1. **Use cp.async** to overlap SMEM init with compute
2. **Use redux.sync.add/min/max** (when type supported) to skip SHFL chain
3. **Vector loads**: `ld.shared.v4` instead of 4× scalar
4. **Welford 1-pass** for LayerNorm (saves 1 reduction tree)
5. **Use mma.sync** for normalization-with-projection patterns

---

For per-task rigor docs and complete commit history, see `M1_V4_DEEP_DIVE_INDEX.md`.
