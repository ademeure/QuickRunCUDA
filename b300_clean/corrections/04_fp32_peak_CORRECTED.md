# 04 — FP32 / FFMA / FP64 / IMAD non-tensor compute peaks (CORRECTED)

Date: 2026-04-22. Cross-references: `04_fp32_peak.md`, `05_fp_precision_nontensor.md`,
`V8_FFMA_PEAK_VERIFIED.md`, `V8_FADD_FMUL_PEAK.md`, `V8_FP64_PEAK_VERIFIED.md`,
`V8_IMAD_PEAK_VERIFIED.md`, `A1_DUAL_ISSUE_RIGOR.md`, `A2_SCHEDULER_RIGOR.md`,
`A4_FFMA_PORT_PRESSURE.md`, `A6_PER_PIPE_REFERENCE.md`, `D6_RF_PORT_RIGOR.md`,
`B1_DUAL_ISSUE_FFMA_IADD3.md`, `B2_FFMA_LDG_DUAL.md`, `V40` / `V49` / `V50`,
`B300_TRUE_REFERENCE.md`, `CLAUDE.md` "B300 SXM6 theoretical peaks" section.

---

## Theoretical (sm_103a, 148 SMs, 128 FP32 cores/SM = 4 SMSP × 32 lanes)

| Op | @ 2032 MHz boost | @ 1920 MHz lock | Source |
|---|---:|---:|---|
| FFMA (FP32 FMA) | **76.96 TFLOPS** | **72.71 TFLOPS** | 148 × 128 × 2 × clk |
| FADD / FMUL (1 FLOP/inst) | 38.48 TFLOPS | 36.36 TFLOPS | half FFMA |
| FP64 DFMA (1:64 of FP32) | **1.203 TFLOPS** | 1.136 TFLOPS | `cudaDeviceGetAttribute SingleToDoublePrecisionPerfRatio = 64` |
| IMAD (32-bit, 1:2 of FP32) | **38.48 Tops** | 36.36 Tops | CUDA C PG Table 13-1 |
| IADD3 (FMA pipe per V40) | 38.48 Tops | 36.36 Tops | 1 inst/SMSP/cy |

## Verified peak measurements

| Op | Clock | Measured | %SoL | Best source |
|---|---|---:|---:|---|
| FFMA 2-source `fma %0,%0,%1,%0` | 2032 boost | **75.2 TFLOPS** | **97.65%** | `V8_FFMA_PEAK_VERIFIED.md` (ncu pipe_fma 97.64%) |
| FFMA NCHAIN=3 rotating + immediate | 2032 boost | **74.62 TFLOPS** | **96.92%** | `B300_TRUE_REFERENCE.md` (commit `06b0d8d`) |
| FFMA definitive (BS=1024 ILP=8) | 2032 boost | 75.92 TFLOPS | 98.6% | `04_fp32_peak.md` `fp32_peak_definitive.cu` |
| FFMA 3-source distinct (port-limited) | 1500 lock | 18.87 TIPS_inst → 27.6 TFLOPS-equiv | ~65% | `A4_FFMA_PORT_PRESSURE.md` / `D6_RF_PORT_RIGOR.md` |
| FFMA locked | **1920 lock** | **62.17 TFLOPS** | **85.5%** | `B300_TRUE_REFERENCE.md` (`e1a1220`) |
| FADD | 2032 boost | 37.4 TFLOPS | 97.65% | `V8_FADD_FMUL_PEAK.md` |
| FMUL | 2032 boost | 37.3 TFLOPS | 97.62% | `V8_FADD_FMUL_PEAK.md` |
| FP64 DFMA | 2032 boost | **1.203 TFLOPS** | **100.00%** | `V8_FP64_PEAK_VERIFIED.md` (ncu pipe_fp64 100%) |
| IMAD 32-bit | 2032 boost | 38.4 Tops | 99.7% | `V8_IMAD_PEAK_VERIFIED.md` |

**Clock paradox** (documented HIGH-confidence): `nvidia-smi -lgc 2032` actually pins
to **1920 MHz**, NOT 2032. Default unlocked sustains 2032 under FFMA load. EVERY peak
claim must annotate which clock is in effect — the ~6% gap between 1920 and 2032
explains most "noisy" historical numbers.

## Pipe placement (V40 ALU ladder, 1500 MHz, % of 1/cy/SMSP per lane)

| Op | Glane/s @ 1500 | %SoL | Pipe |
|---|---:|---:|---|
| FADD / FFMA / **IADD3** | 25-26 | 67% | **FMA pipe (shared)** |
| LOP3 / IMUL | 18.7 | 48% | INT-bit (half rate) |
| PRMT | 13.9 | 36% | permute |
| ISETP | 8.4 | 22% | compare |

**IADD3 lives on the FMA pipe** (V40, commit `d1d09c5`), NOT a separate INT pipe.
This is consistent with the unified-cluster model in A1/A6: "Cluster A: FFMA + IMAD
+ IADD3". Earlier wording in A6 calls IADD3 "ALU (unified)" which conflated the
naming with LOP3's INT-bit pipe — they are physically distinct sub-pipes inside the
unified-cluster dispatch.

## Dual-issue: per V49/V50, capped well below "perfect"

| Pattern | Glane/s | Overlap factor | Source |
|---|---:|---:|---|
| FFMA solo | 26066 | — | V49 |
| LOP3 solo | 18578 | — | V49 |
| FFMA + LOP3 same warp | 24336 | **55%** | V49 (`501134a`) |
| FFMA + IADD3 same warp | 27969 | **54%** | V49 |
| FFMA + PRMT same warp | 20303 | **51%** | V49 |
| FFMA + LOP3 warp-specialized (4+4 warps) | 16724 total | **74%** of separate-pipe theoretical | V50 (`fbe1c18`) |
| FFMA + IADD3 same warp, NC=8 | — | 17% | B1 |
| FFMA + LDG (chain) | — | 1% | B2 |
| FFMA + LDG (no chain) | — | 12% | B2 |
| FFMA + SHFL | — | 14.7% | A6 |
| FFMA + MUFU | — | ~100% | A6 / `8012b98` |

## RETRACTIONS

1. **"154 TFLOPS FP32" / "256 FP32 cores/SM" / "FP32 dual-issue doubles rate"** — RETRACTED.
   B300 has 128 FP32 cores per SM (same as Hopper). True peak = 76.96 TFLOPS @ 2032.
   The 154 number is a 2× formula error. (Already retired in `04_fp32_peak.md`
   "RETIRED claims" table; re-asserted here.)

2. **"FP32 FFMA peak 71.8 TFLOPS = 93%"** (`05_fp_precision_nontensor.md` §1) —
   PARTIALLY SUPERSEDED. 71.8 was measured at 1920 MHz (= 98.7% of 72.71). When
   restated against 2032-MHz theoretical it shows as 93%, which understates the
   pipe efficiency. Use **74.6-75.9 TFLOPS = 97-99%** (`V8_FFMA_PEAK_VERIFIED.md` /
   `04_fp32_peak.md`) as the canonical 2032 number; cite 71.8 only with explicit
   "@ 1920 MHz" annotation.

3. **"FFMA latency 23 cy"** (legacy catalog l.19565, AUDIT_NOTES l.53) — RETRACTED.
   Real latency is 4.019 cy; 23 was the empty-loop BRA floor measured at the same
   time and misattributed (see A1's empty-loop floor lesson).

4. **"Self-op `FFMA Ra,Ra,Ra,RZ` is 2× slower from RF port pressure"** (catalog l.1947,
   "8.46 cy") — RETRACTED. On B300 self-op = 4.02 cy, identical to diff-src; reuse
   cache makes 1- and 2-unique-source patterns equivalent. (`04_fp32_peak.md` retired
   table.)

5. **"FP32 FMA = 38 TFLOPS at 386 W"** (catalog ~l.17914) — RETRACTED. Estimate from
   under-saturated kernel; real peak 74.6 TFLOPS @ 361 W (`peak_ffma_power.cu`).

6. **"FP64 DFMA 0.95 TFLOPS"** (catalog l.35) — SUPERSEDED. 1.20 TFLOPS @ 2032 MHz
   with ≥4 warps/SM is the true peak (= 100% of 1:64 ratio, `V8_FP64_PEAK_VERIFIED.md`).
   0.95 was 1920 MHz + insufficient warps.

7. **"IMAD has same throughput as FP32 FFMA"** (initial assumption corrected in
   `V8_IMAD_PEAK_VERIFIED.md`) — RETRACTED. IMAD is 1:2 of FP32 per CUDA PG Table
   13-1; peak 38.5 Tops, NOT 76.97 Tops.

8. **"HFMA2 = 308 TFLOPS-FP16 = 2× FFMA FLOPS"** (catalog l.5096) — RETRACTED at
   l.6390. Packed FP16/BF16 FMA hits the SAME pipe-FLOPS ceiling as FP32 (~72-75
   TFLOPS chip). No 2× speedup outside tensor cores.

9. **"FFMA + IADD3 cleanly dual-issue on separate pipes"** (early NINJA / catalog
   `a0bde33`, `f578755`) — RETRACTED. V49 measures 54-55% same-warp efficiency
   even with warp-specialization only reaching 74%. Use the V49/V50 numbers.

10. **"58.6 TFLOPS = 76.2% peak"** (AUDIT_NOTES / `fp32_peak2.cu`) — RETRACTED.
    CPU `std::chrono` timing including launch overhead, possibly on throttled GPU 0.
    Real peak 75.9 TFLOPS.

## UNRESOLVED

1. **Why is dual-issue capped at 55% same-warp / 74% warp-spec when FMA and INT
   pipes are physically separate?** V49 attributes it to "warp scheduler dispatch
   slot is shared (4 inst/cy/SM total)". A2 separately demonstrated SMSP issue port
   sharing (MUFU warp slows its sibling FFMA warp on same SMSP). Hypothesis: each
   SMSP issues exactly 1 warp-inst/cycle regardless of pipe diversity, and only when
   the consumer pipe takes ≥2 cy/inst (MUFU @ 4 cy) does FFMA fit in the gaps. Needs
   direct ncu `smsp__inst_issued.avg.per_cycle_active` measurement under the V49
   patterns to confirm.

2. **A6 "FMA pipe at 0.66 inst/SMSP/cy" vs V8 "97.64% pipe_fma"** — A6 measures at
   2 warps/SMSP and 1500 MHz; V8 at 8 warps/SMSP and 2032 MHz. The gap is occupancy,
   not pipe physics. A6 hypothesised "need 4+ warps/SMSP for 98%"; V8 confirms.
   Document both regimes; do not cite A6's 0.66 as a ceiling.

3. **D6 says 3-distinct-source FFMA caps at 65% (one RF read port short).** A4
   independently reproduces. But `B300_TRUE_REFERENCE.md` headline FFMA 74.62 uses
   "NCHAIN=3 rotating + immediate constant" — only 2 distinct register sources
   (the immediate avoids a 3rd register read). All "near-peak" recipes implicitly
   sidestep the port limit. A worst-case all-distinct outer-product GEMM kernel
   would hit ~50 TFLOPS, NOT 75. Catalog should flag this.

4. **`05_fp_precision_nontensor.md` §1 line "FFMA peak 71.8 TFLOPS = 93% of 76.96"**
   mixes the 1920-MHz measurement with the 2032-MHz denominator. This is a clock-state
   bookkeeping bug. Either re-state as "62.17 TFLOPS @ 1920 = 85.5%" (matching
   `B300_TRUE_REFERENCE.md`) or re-measure at 2032 to get the 97-99% number.

5. **Multiple "best FFMA" measurements disagree by ~1 TFLOPS:**
   `V8_FFMA_PEAK_VERIFIED.md` 75.2 / 97.64% (NCHAIN=8, immediate constant)
   `04_fp32_peak.md` 75.9 / 98.7% (`fp32_peak_definitive.cu`, ILP=8 BS=1024)
   `B300_TRUE_REFERENCE.md` 74.62 / 96.92% (NCHAIN=3 rotating + immediate)
   These are within measurement noise (~1.5%) and all 97-99% of theoretical, but the
   canonical headline number should be picked. Recommend **75.9 TFLOPS = 98.6%** as
   the headline (highest, definitive kernel, SASS-verified, ncu-confirmed).
