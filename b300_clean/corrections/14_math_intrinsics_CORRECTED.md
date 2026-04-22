# 14 Math Intrinsics — CORRECTED reference (MUFU / SHFL / REDUX)

Date: 2026-04-22. Sources cross-checked: `14_math_intrinsics.md`,
`V8_MUFU_PEAK.md`, `V8_SHFL_PEAK.md`, `Q3_WARP_REDUCE_RECIPES.md`,
`CHAIN_FP_MUFU_LATENCY.md`, `V41_V48_FINDINGS.md`, `M16_V9_FULL_SYNTHESIS.md`,
`B300_TRUE_REFERENCE.md`. GPU: B300 SXM6 sm_103a, 148 SMs.

---

## 1. MUFU per-op throughput AND latency (separate columns)

| PTX op | SASS | Chain latency (cy) | Throughput (Gops/s, chip) | % of 1/(4cy)/SMSP SoL | Notes |
|---|---|---:|---:|---:|---|
| `ex2.approx.f32` | `MUFU.EX2` | **14.14** | **9.22** | **95.8%** | ANOMALY: 2x faster than all other MUFU |
| `lg2.approx.ftz.f32` | `MUFU.LG2` | 18 | 4.74 | 49% | half-rate |
| `rcp.approx.f32` | `MUFU.RCP` | **42.10** | 4.74 | 49% | longest pure-pipe latency |
| `rsqrt.approx.ftz.f32` | `MUFU.RSQ` | 18 (ftz) / **40.10** (rn) | 4.74 | 49% | |
| `sqrt.approx.f32` | `MUFU.SQRT` | 18 / 40 | 4.74 | 49% | |
| `sin.approx.f32` | `MUFU.SIN` | **24.02** | 4.74 | 49% | |
| `cos.approx.f32` | `MUFU.COS` | 24 | 4.74 | 49% | |
| `tanh.approx.f32` | `MUFU.TANH` | 18 | ~4.7 | 49% | |

**The EX2 anomaly is real and isolated.** V41 (the per-op MUFU sweep) measured
EX2 at 95.8% of the 1/(4cy)/SMSP SoL while every other MUFU op sat at ~49%.
This contradicts the original `14_math_intrinsics.md` table, which lists every
MUFU at "0.5 issue/SMSP/cy" with EX2 only "0.5-0.63" (a slight bump). The real
gap is 2x, not 1.26x.

### EX2 latency-vs-throughput note (CHAIN_FP_MUFU_LATENCY)

EX2 has split latencies: pure EX2->EX2 chain runs at 14 cy (writeback shortcut),
but FFMA->EX2->FFMA pays an extra ~22 cy per cross-pipe round-trip. RCP does
NOT show this split (FFMA->RCP->FFMA = 50 cy = perfect linear sum). When
designing softmax-style kernels, do not assume EX2 latency is 14 cy if the
result feeds FFMA — budget closer to ~30 cy.

---

## 2. SHFL vs REDUX — raw rate vs algorithm rate

| Primitive | Latency (cy, chained) | Raw throughput (Telements/s, chip) | Notes |
|---|---:|---:|---|
| `SHFL.BFLY` (warp shuffle) | ~5 (chain) | **9.48** (V38) | 1 inst per (4 cy) per SMSP |
| `redux.sync.{add,min,max,and,or,xor}.u32/s32` | ~11 | **9.09** (V37) | 1 inst per (4 cy) per SMSP |

**Raw rates are equal** (within measurement noise). Both share the same MIO /
shuffle pipe at 1 inst per 4 cy per SMSP.

### Reconciling the "REDUX 4x SHFL" claim

The phrase "redux.sync.min/max 4x SHFL" appears in the user MEMORY entry for
the V4 loop, and `V8_SHFL_PEAK.md` line 30 cites "4x SHFL per V4 prior
findings". The actual measurement (`Q3_WARP_REDUCE_RECIPES.md`) shows
**REDUX.SUM = 11.61 cy vs 5-step SHFL chain = 27.19 cy = 2.34x speedup**, not
4x. The speedup is ALGORITHM-LEVEL (one REDUX replaces a 5-step SHFL+ADD tree),
NOT a per-instruction throughput advantage.

So:
- Raw inst/cy: SHFL == REDUX (V37, V38).
- Algorithm-level reduce-32-lanes: REDUX 2.34x faster than SHFL chain.
- Source for the "4x" number: not located. Treat "4x" as legend; cite 2.34x
  with `Q3_WARP_REDUCE_RECIPES.md` as the authoritative measurement.

`V8_SHFL_PEAK.md` itself reports only 3 G warp-SHFL/s (= ~96 G thread-SHFL/s),
which is FAR below the 9.48 Telements/s headline of V38. The discrepancy is
units and ILP: V8's 3 G is **warp-SHFL/s with chain-dep self-feed** (latency
bound, ~100 cy/SHFL effective), while V38's 9.48 Telements/s is per-thread,
ILP-saturated, throughput bound. Both can coexist; just label clearly.

---

## 3. Latency ladder (MUFU + warp ops, single chain)

| Op | Latency cy | ns @ 2.032 GHz |
|---|---:|---:|
| FFMA / FADD / FMUL / HFMA2 | 4.04-4.22 | 2.0-2.1 |
| MUFU.EX2 (chained EX2->EX2) | 14.14 | 7.0 |
| MUFU.EX2 (cross-pipe FFMA->EX2->FFMA) | ~30 | ~15 |
| MUFU.LG2 / SQRT / RSQ.ftz / TANH | 18 | 8.9 |
| MUFU.SIN / COS | 24.02 | 11.8 |
| MUFU.RSQ / SQRT (non-ftz IEEE) | 40.10 | 19.7 |
| MUFU.RCP | 42.10 | 20.7 |
| `redux.sync.add.u32` | ~11.6 | 5.7 |
| `SHFL.BFLY` (chain) | ~5.4 | 2.7 |

---

## 4. Throughput ceiling reconciliation

| Source | Claim | Verdict |
|---|---|---|
| `14_math_intrinsics.md` sec 1 | "MUFU peak ~4.8 TGOps/s; EX2 1.7x faster ~8.1" | LOW: EX2 is 2x not 1.7x, others = 4.74 (matches), all per V41 |
| `V8_MUFU_PEAK.md` | "47.8 G thread-MUFU/s at 99.5% XU pipe util (rsqrt)" | OK at 1500 MHz / 1-chain regime; do not compare to 9.62 T |
| `M16_V9_FULL_SYNTHESIS.md` | XU peak "47.8 GMUFU/s @ 99.5%" | Correct as a 1-chain figure, MISLEADING as "the" XU peak. True saturated MUFU = 4.74 Gops/s/chip = 4740 GMUFU/s |
| `V41_V48_FINDINGS.md` | EX2 9.22 / others 4.74 Gops/s | AUTHORITATIVE — adopt |

The 100x gap between V8 (47.8 G) and V41 (4740 G) is the textbook
ILP-saturation gap: V8 used self-dep chain on `rsqrt` (latency-bound),
V41 used independent MUFU streams (throughput-bound).

---

## RETRACTIONS

1. **`14_math_intrinsics.md` sec 1**: "EX2 is 1.7x faster (~8.1 TGOps/s)" —
   RETRACT. Correct: 2x faster. EX2 = 9.22 Gops/s, others = 4.74 Gops/s.
2. **`V8_SHFL_PEAK.md` line 30**: "redux.sync 4x SHFL on B300" — RETRACT
   the "4x". Replace with "REDUX 2.34x faster than 5-step SHFL chain
   (algorithm-level); per-instruction rates equal".
3. **`M16_V9_FULL_SYNTHESIS.md` table I**: "XU 47.8 GMUFU/s" as the canonical
   peak — RETRACT framing. It is a single-chain rsqrt latency-bound number.
   True throughput-bound XU peak is ~4.74 Gops/s/chip (most MUFU) /
   9.22 Gops/s (EX2). Keep 47.8 G as a "1-chain rsqrt" footnote.
4. **`14_math_intrinsics.md` sec 7 per-SM table**: "exp2f 34.9 / sqrt-rsqrt
   22.5 / sin-cos 20.6 Gops/s/SM" — INCONSISTENT with V41 (which says all
   non-EX2 MUFU are equal at 4.74 Gops/s/chip = 32 Gops/s/SM, and EX2 = 62
   Gops/s/SM). Mark sec 7 as LOW confidence pending re-measurement; V41 is
   the higher-rigor source.
5. **User MEMORY note "redux.sync.min/max 4x SHFL"** — origin unknown,
   no measurement file backs the 4x figure. Treat as folklore; cite 2.34x.

---

## UNRESOLVED

1. **Why is EX2 2x faster than every other MUFU op on B300?**
   - Hypothesis A: dedicated EX2 hardware path (separate sub-pipe within XU),
     possibly because EX2 underlies `expf` / `tanhf` / `__expf` / softmax —
     the most common transcendental in ML workloads. NVIDIA may have widened
     this lane to 1/(2cy)/SMSP while keeping LG2/RCP/etc at 1/(4cy).
   - Hypothesis B: EX2 algorithm uses fewer internal correction iterations
     (smaller polynomial). RCP/RSQRT/SQRT need Newton-Raphson refinement;
     EX2 is a direct reduce-and-table lookup with one polish FMA.
   - Hypothesis C: EX2 has a separate writeback port that LG2/SIN/COS share.
   - Hopper (sm_90) reportedly does NOT have this 2x gap — needs verification.
   - Test to discriminate: SASS-count internal `MUFU.EX2` vs `MUFU.LG2` micro-ops,
     and run mixed `EX2 + LG2` 50/50 chain — if both peak at 4.74 G summed,
     they share a port; if they sum to 9.48 G, EX2 has its own.

2. **EX2 split latency mechanism** (14 cy chained, ~30 cy cross-pipe):
   need a longer FFMA-EX2 chain (NC=8+) to extract the result-availability
   latency separately from issue interval. Currently best-fit hypothesis only.

3. **Per-SMSP MUFU conflict behavior**: the catalog says "0.5/SMSP/cy" but
   does not isolate what happens when all 4 SMSPs issue MUFU same cycle.

4. **Why SHFL self-dep chain measures only 3 G warp/s in V8** while V38 hits
   9.48 Telements/s — confirmed to be ILP/latency vs throughput, but the
   exact chain-latency of SHFL.BFLY (5 cy assumed) needs a clean rigor test.

5. **`tanhf` actual throughput**: catalog's per-SM table claims 22.5 Gops/s/SM,
   but V41 grouping puts TANH at 4.74 Gops/s/chip = 32 Gops/s/SM. Discrepancy
   unresolved; V41 has the rigor edge.

6. **REDUX latency 11.6 cy origin**: is this 1-cycle issue + ~10 cy reduce-tree
   latency (HW), or a serialized 5-step internal? No microbench file has
   isolated this; only Q3 chain throughput.
