# Report #7 — Final SFU pipe model (both sub-agents converge)

Both independent sub-agent investigations (no shared context) plus my own
follow-up experiments converge on a clean, testable architectural model for
Blackwell's SFU.

## The definitive rule

**An F2FP/SFU instruction can dual-issue to 64 thread-inst/SM/clk iff its SASS
opcode encoding uses exactly 1 regfile read port. Instructions encoding ≥2
regfile read ports cap at 32/SM/clk (single-slot).** Writeback width, MERGE_C
semantics, pipeline latency, direction, and destination-register disjointness
are all *irrelevant*; only the **read-port footprint of the opcode encoding**
matters.

This was proven by the second sub-agent's `bench_f2fp_port_hypothesis.cu`
(V10–V19), which constructed variants specifically to break each alternative
hypothesis. The controls:

| Test | Disproves |
|---|---|
| V2 `PACK_AB` (2 reads, 32-bit writeback, no MERGE_C) = 32/clk | "MERGE_C dest-read" (V2 has no MERGE_C) |
| V13/V17 (2 and 4 independent packs to disjoint dest regs) = 32/clk each | "writeback-bus conflict" (disjoint dests can't dual-issue either) |
| V4 `PACK_B tf32` with full saturation+rounding = 64/clk | "pipeline pitch" (rounding doesn't slow it) |
| Both directions have 32- and 64-capable members (V1 unpack 64, V2 pack_ab 32; V4 pack_b 64) | "direction-specific subunit" |

**Additional refinement (sub-agent 2 discovery):** every emitted
`UNPACK_B_MERGE_C` has `Rc = RZ` in SASS — ptxas knows the upper-16 of the
narrow output is don't-care, so MERGE_C isn't even *reading* the destination.
But the opcode still reserves the read-port slot, which is what locks out
the second issue.

## Read-port assignment table (all SASS opcodes measured)

| SASS | Read ports | Solo /SM/clk | Class |
|---|:---:|---:|---|
| `F2FP.F16.E4M3.UNPACK_B` (unpack narrow) | **1** | **64** | dual-issue ✓ |
| `F2FP.F16.E5M2.UNPACK_B` | 1 | 64 | dual-issue ✓ |
| `F2FP.F16.E2M1.UNPACK_B` (FP4 unpack) | 1 | 64 | dual-issue ✓ |
| `F2FP.F16.E2M3.UNPACK_B` (FP6 unpack) | 1 | 64 | dual-issue ✓ |
| `F2FP.F16.E3M2.UNPACK_B` (FP6 unpack) | 1 | 64 | dual-issue ✓ |
| `F2FP.BF16.E8.UNPACK_B` (ue8m0 → bf16x2) | 1 | 64 | dual-issue ✓ |
| `F2FP.SATFINITE.TF32.F32.PACK_B` (f32→tf32) | 1 | 64 | dual-issue ✓ |
| `F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C` (pack narrow) | 2 (incl. MERGE_C) | 32 | single-slot |
| `F2FP.F16.F32.PACK_AB` (f32→f16x2) | 2 | 32 | single-slot |
| `F2FP.BF16.F32.PACK_AB` (f32→bf16x2) | 2 | 32 | single-slot |
| `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` | 3 | 31 | ~2-3% penalty |
| `F2FP.SATFINITE.F16.F32.MERGE_C` (scalar pack) | 2 | 24 | special slow path |
| `MUFU.EX2` | 1 | 32 | SFU, 1 slot but half-rate pipe |
| `MUFU.RSQ`/`SIN`/`COS`/`LG2`/`TANH`/`SQRT` | — | 16 | 2 slots, slower |
| `POPC.B32` | — | 16 | 2 slots |
| `F2F.F16.F32` (non-sat scalar) | — | 16 | 2 slots |

Bonus finding: `F2FP.F16.F32.PACK_B` (tf32 with 1 source) hits 64/clk even
with full saturation+rounding — proving the saturation/rounding logic
**isn't** the issue. The single-read encoding alone unlocks dual-issue.

## Complete SM pipe map

Combining all findings (my work + both sub-agents):

### 2 SFU issue slots per SM/cycle

- Each "slot" can host 1 F2FP/MUFU/F2F/POPC op per cycle
- Ops with ≥2 read ports take both slots (single-issue → half rate)

### Separate "FMA pipe" shared by:
- FFMA, FMUL, FADD (FP32)
- **IMAD** (integer multiply-add — confirmed NOT on INT pipe; contends with FFMA per cross-matrix Report #6)

### Separate "INT-logic pipe":
- LOP3.LUT (xor, and, or, select)
- IADD3, SHL, BREV, etc.

### Separate "FP16x2 FMA pipe":
- HFMA2, HADD2 (independent of FP32 and SFU)

### Separate "Tensor-core pipe":
- HMMA.*, BMMA.* (uses SM dispatch but own functional unit)

### Memory pipe (LSU):
- LDG, STG, LDS, STS — separate from compute pipes
- Affected by address patterns (cache-line coherence across blocks, coalescing)

### SMSP-level dispatch contention
At saturation, all compute ops compete for the 4-per-SM warp scheduler
dispatch slots. This creates "apparent" contention between otherwise-
independent pipes (e.g., FFMA vs F2FP) when both are near their peak. This
is NOT pipe-sharing — it's dispatch bandwidth exhaustion.

## What makes F2FP quantization kernels fast or slow

### Fast — these are effectively free additions to an F2FP-bound kernel:
- Integer logic: LOP3, IADD3, SHL for address math
- HFMA2/HADD2 for f16x2 scaling math
- `st.local.*` (stack spills) — compiled to registers, no actual store

### Slow — these directly reduce F2FP throughput:
- **MUFU.EX2 and other transcendentals** (softmax, GELU approx)
- **Scalar f32→f16 without `.satfinite`** (hits F2F pipe, 16/clk max)
- **Multiple STGs per iter when cache lines collide across blocks** — 1 STG
  can halve throughput at 592 blocks writing to same region
- **Atomics** — 1 atom.global essentially freezes the kernel

### Cliffs to avoid:
- **STG cliff at ~4 per 32 F2FPs** — store-queue saturation, F2FP drops to 14%
- **More than 16 HMMAs per pack** — dispatch saturation
- **Divergent warps** — F2FP rate scales linearly with active lanes

## Practical takeaway for kernel writers

```
# Ideal quant-pack iteration on Blackwell:
for chunk:
  load_f16x2     [LSU]    # once per chunk, wide (v4.f32)
  scale_f16x2    [FMA]    # FFMA or HFMA2 — free
  pack_narrow    [SFU]    # F2FP PACK_AB_MERGE_C @ 32/clk = bottleneck
  pack_narrow                                       @ 32/clk
  ...(8-16 packs per iter — interleave with integer addr-math)
  # Do *not* add MUFU, F2F, or scalar CVTs here!
  store          [LSU]    # 1 wide STG per iter (at cliff threshold)
```

Expected throughput per SM: **32 pack-instructions × 2 elements = 64
elements/SM/clock** = 19 GElements/sec per SM × 148 SMs × 2.032 GHz =
**19 T narrow-elements/sec** on B300 (theoretical peak for pack-bound kernel).

For dequant (unpack only), expected:
- 64 unpack-instructions × 2 elements = 128 elements/SM/clock = **38 T
  elements/sec**. Almost always HBM-bound first (8.2 TB/s ≈ 4 T f16/sec
  read, 4 T f16 dequant output possible).
