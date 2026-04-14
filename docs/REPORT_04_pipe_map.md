# Report #4 — Complete Blackwell SM Instruction Pipe Map

Combining my and sub-agent's (independently designed) findings. Sub-agent's full
writeup in `/tmp/subagent_findings_1.md` (also in this dir). Strong agreement
between the two independent investigations.

## The "2-slot SFU" model — validated across 20+ instructions

Every instruction we've measured fits one of four pipe classes:

### SFU (2 issue slots per SM per cycle)

| Instruction | Solo /SM/clk | Slot cost | F2FP contention at N=32 |
|---|---:|---:|---:|
| `F2FP.F16.E4M3.UNPACK_B` (unpack narrow) | **64** | **½** (dual-issues) | — (saturates alone) |
| `F2FP.E4M3.F16.UNPACK_B_MERGE_C` (pack narrow) | 32 | 1 | — |
| `F2FP.F16.F32.PACK_AB` (f32→f16x2) | 32 | 1 | — |
| `F2FP.SATFINITE.TF32.F32.PACK_B` (tf32) | **64** | **½** | — |
| `MUFU.EX2` | 32 | 1 | F2FP drops to 15.27 (drops 3/4) |
| `MUFU.RSQ`/`SIN`/`COS`/`TANH`/`LG2`/`SQRT` | 16 | 2 | huge drop |
| `F2F.F16.F32` / `F2F.BF16.F32` (non-sat scalar) | 16 | 2 | huge drop |
| `POPC.b32` | 16 | 2 | huge drop |

All of these share the SFU's 2-issue-slots-per-cycle budget. In mixes the rule
is approximately:

    Σ(slot_cost × op_count) ≤ 32 slot-ops/SM/cycle

Verified for EX2+RSQ (Report #1) and pack+unpack interleaving (REPORT_01).

### FMA pipe (128 /SM/clk)

- `FFMA.f32`, `FMUL.f32`, `FADD.f32`
- `IMAD` (integer multiply-add — on Blackwell it's on the FMA pipe, not INT)

F2FP contention at N=32: **none measurable** (≤3% drop, consistent with IPC
noise).

### INT / Logic pipe (~128 /SM/clk)

- `LOP3.LUT` (xor, and, or)
- `IADD3`, `SHL`
- `ISETP`, etc.

F2FP contention: **zero** (IADD3, SHL, IADD; LOP3 a tiny 3% at extreme).

### Half-precision FMA pipe (separate from FP32)

- `HFMA2.rn.f16x2` / `HADD2.f32`

F2FP contention: **zero** at all N_COMP tested.

### "SMSP dispatch port" (warp-scheduler level sharing, not execution)

- `SHFL.*` (warp shuffle network) — ≈9600 /SM/clk alone
- `HMMA.*` (tensor-core MMA) — affects FFMA the same way; not F2FP-specific
- LDG / LDS / STG / STS — memory pipe proper

These don't share the SFU *execution* unit, but at high density they steal
warp-scheduler cycles that the SM would otherwise use to dispatch F2FP.
Symptom: F2FP throughput drops, but generic FP32 and INT throughput drop by
the same factor.

### Memory pipe (LSU)

- LDG at low count: nearly free (<5% drop per LDG at N=4)
- LDG at N=32 cached: ~35% F2FP drop
- LDG at N=32 uncached: ~55% F2FP drop (cache-miss penalty adds latency)
- **STG contention is address-pattern dependent**:
  - **Per-block-distinct coalesced addresses**: mild, linear
  - **Inter-block cache-line collisions**: catastrophic, 1 STG halves F2FP
  - **Uncoalesced per-thread strided writes**: even worse

## Per-op solo throughput reference (all SASS verified)

| SASS | Units/SM/clk | Slot interpretation |
|---|---:|---|
| F2FP unpack | 64 | dual-issue SFU (½ slot) |
| F2FP pack (MERGE_C or PACK_AB) | 32 | 1 SFU slot |
| F2FP scalar MERGE_C | 24 | 1 SFU slot + extra latency |
| MUFU.EX2 | 32 | 1 SFU slot |
| MUFU.RSQ/SIN/... | 16 | 2 SFU slots |
| POPC.B32 | 16 | 2 SFU slots |
| F2F.F16.F32 (non-sat) | 16 | 2 SFU slots |
| SHFL.BFLY | 32 | 1 SMSP dispatch slot, separate data path |
| LDG.E | ~10-60 | memory pipe |
| LDS.B32 | 87 | shared-mem pipe |
| FFMA | 128 | FP32 FMA pipe |
| HFMA2 | 128 | FP16x2 FMA pipe |

## Practical implications

1. **NVFP4/FP8 quant kernels that softmax/GELU in the same loop** lose
   throughput because softmax uses `MUFU.EX2` (1 SFU slot) which competes with
   F2FP pack (1 SFU slot). A fused quant-softmax kernel at 1:1 op ratio runs at
   32 F2FPs/SM/clk + 16 EX2/SM/clk = worse than either alone.
2. **Dequantize-only kernels** can run unpack at 64/SM/clk = 128 elements/SM/clk,
   often HBM-bound well before SFU is a concern.
3. **Mix with FP32 math (FFMA, FMUL) is free** — the two pipes are truly
   independent. This is the "right" instruction mix for quantization kernels.
4. **POPC on the SFU is a gotcha** — using it in quant loops (e.g. for signed
   mantissa tracking) kills F2FP throughput.
5. **HMMA + F2FP** fight for the warp-scheduler dispatch port, but this is not
   specific to F2FP — any instruction from any pipe drops under HMMA pressure.

## Side findings on corner cases

- `cvt.rn.f16.f32` without `.satfinite` silently becomes `F2F.F16.F32` +
  `HADD2.F32` widen — ~2-3× slower than `.satfinite` path. **Always use
  `.satfinite` for scalar f32→f16/bf16.**
- `cvt.rna.tf32.f32` is software-emulated (~12 IMAD/LOP3 chain) at 3/SM/clk.
  `cvt.rn.tf32.f32` is hardware F2FP at 64/SM/clk. **Use `.rn` unless you
  specifically need round-to-nearest-away.**
- `cvt.f32.f16` / `cvt.rn.f32.bf16` widening is free (compiler-elided as
  register-bit move).
- `cvt.rs.*.x4.f32` is **not** a single HW instruction — ptxas expands it to 2
  x2 F2FP ops.

## Independent investigation agreement summary

Sub-agent (no prior knowledge of my findings) and I converged on:
- 2-slot SFU budget model (both)
- MUFU + F2FP share SFU (both)
- FFMA/IMAD/FMUL/LOP3 independent from F2FP (both)
- Pack 32 vs Unpack 64 (both)
- HMMA contention is dispatch-level not SFU-level (sub-agent only — I hadn't tested)
- POPC on SFU (sub-agent only — new!)
- F2F.F16.F32 on SFU (sub-agent confirmed; I measured latency only)

## Open questions / next steps

- **Tensor-core descriptor setup ops** (`utmaldg`, `cp.async.bulk.tensor`) —
  do they also use dispatch slots?
- **Warp divergence impact on F2FP** — does a partial warp execute F2FP at
  same rate?
- **Across-SM effects** — is there any SM-to-SM scheduler coupling at very
  high aggregate F2FP load?
- **The "pack 32 vs unpack 64" mystery**: why is pack limited to 1 SFU slot
  when the slot count is per-cycle? Is there actually a 2nd pack lane that's
  blocked by MERGE_C internal forwarding? **Hypothesis**: the pack issues a
  byte-wide write-back into the low half of its destination, which shares the
  writeback bus with another pack; only one pack-write-back can commit per
  cycle. Unpack writes full 32-bit into a fresh register, going to a dedicated
  writeback lane.
