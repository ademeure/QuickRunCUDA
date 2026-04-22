# A1-D6 Inconsistency Log

Date: 2026-04-22. Wave-3 swarm.

Specific number-on-number inconsistencies between A/B/C/D rigor docs
(2026-04-20 timeframe) and V40-V51 (2026-04-21..22). Companion to
`A_TO_D_RIGOR_AUDIT.md`.

Format: claim — source (file:loc) — conflicting value — source — verdict.

---

## Dual-issue overlap (multi-doc)

| Pattern | A/B/C/D claim | V49/V50 claim | Verdict |
|---|---|---|---|
| FFMA + LOP3 same-warp | A1 = "6.5% overlap" (line 27) | V49 = **55%** (24336 Glane/s vs sum 44644) | **A1 superseded**. V49 used OP=3 multi-warp; A1 was N=16 unrolled OP=1, single warp. |
| FFMA + IADD3 same-warp | A1 = "**14% WORSE** than serial (cluster-A contention)" (line 28); B1 = "17% overlap" (line 32) | V49 = **54%** (27969 Glane/s) | **A1+B1 superseded.** V49 deeper ILP + multi-warp + cleaner anti-DCE. The "cluster-A contention" reading was wrong. |
| FFMA + LOP3 warp-spec | A6 implies 4 inst/cy/SM dispatch limit applies | V50 = **74%** of separate-pipe theoretical (16724 total Glane/s) | A6 model is the right shape but doesn't quantify; V50 measures. |
| FFMA + MUFU | A6 ≈100% (commit 8012b98) | Not retested in V40-V51 | A6 unchanged. |
| FFMA + LDG (chain) | B2 = 1% | Not retested | B2 unchanged. |
| FFMA + SHFL | A6 = 14.7% | V37/V38 measured SHFL pipe but not the dual | A6 unchanged for the number; mechanism unverified. |

---

## FFMA chip throughput

| Source | Claim | Clock |
|---|---|---|
| A6 line 80-90 | "FFMA only hits **0.66/SMSP/cy**, not 1.0; need 4+ warps/SMSP for 98%" | 1500 lock, 2 warps/SMSP |
| B1 line 12 | "FFMA = 18.87 TIPS_inst, only 66% of theoretical 28.4" | 1500 lock, 2 warps/SMSP |
| `04_fp32_peak_CORRECTED.md` | FFMA chip peak **75.2 TFLOPS = 97.65%** (V8) | 2032 boost, 8 warps/SMSP |
| `04_fp32_peak_CORRECTED.md` | FFMA at 1920 lock = **62.17 TFLOPS = 85.5%** | 1920 lock |
| V40 | FFMA = 25-26 Glane/s = **67%** | 1500 lock, full persistent |

**Verdict**: All measurements correct under their conditions. A6 + B1
**0.66/SMSP/cy is NOT a chip ceiling** — it's an under-saturated 2-warp
floor. V8 + V49 reach 97-98% with 8 warps/SMSP at 2032 boost. Cite the
clock + warp count whenever quoting FFMA throughput.

---

## ALU pipe ladder (per-instruction)

| Op | A6 inst/SMSP/cy | V40 % of FMA-pipe SoL | Match? |
|---|---|---|---|
| FFMA | 0.66 | 67% | YES (0.66 ≈ 67%) |
| IADD3 | 0.50 | **67%** (FMA tier) | **MISMATCH** — A6 says half, V40 says FMA tier |
| LOP3 | 0.50 | 48% (INT-bit tier) | YES |
| SHF | 0.50 | (in INT-bit) | YES |
| PRMT | 0.50 | **36%** (permute tier) | **MISMATCH** — A6 says 0.50, V40 says 0.36 |
| BFI | 0.46 | (not in V40 ladder) | A6 only |
| BFE | 0.25 (XU) | (not in V40 ladder) | A6 only |
| ISETP | (not in A6) | **22%** (compare tier) | V40 only |
| BREV/POPC/CLZ | 0.125 | (not in V40 ladder) | A6 only |
| SHFL.IDX | 0.25 | (V37/V38 = shuffle pipe, ~0.25) | YES |

**Verdict**:
- IADD3 mismatch: A6's 0.50 from `bench_iadd3_throughput.cu` (2 warps/SMSP).
  V40's 67% from full persistent. Likely warp-count effect; needs A6-style
  re-sweep. Listed as UNRESOLVED #2 in `A_TO_D_RIGOR_AUDIT.md`.
- PRMT mismatch: A6 says 0.50/SMSP/cy (line 19); V40 measures 36%
  (= 0.36/SMSP/cy). `15_integer_bit_ops_CORRECTED.md` UNRESOLVED #2 also
  flags this; likely different ILP / op-mix conditions. Open.

---

## RF read ports (FFMA 3-distinct sources)

| Source | Claim |
|---|---|
| A4 line 13 | 3-unique = 0.61/SMSP/cy (37% slower) |
| D6 line 22 | 3-unique = 1.53 cy/fma = 0.65/SMSP/cy |
| `V10_FMA_SOURCE_COUNT` (per FFMA agent) | 2-source 75.2 TFLOPS, 3-source 51.3 TFLOPS, ratio 0.683 (close to 2/3) |

**Verdict**: All consistent (within 5%). 2-RF-port + reuse-cache theory
holds. A4 and D6 correctly diagnose the same effect; V10 confirms at chip
scale. Per FFMA agent finding "ALL near-peak FFMA recipes use ≤2 unique
register sources" is supported.

---

## SMEM bank conflict (32-way)

| Source | Cost | Regime |
|---|---|---|
| D5 line 18 | **5.74×** (74.77 vs 13.03 cy) | single-warp, latency-bound |
| V44 (per V41-48 doc) | ~2× | latency-bound chain-serial |
| V45 (per V41-48 doc) | **~1× (free)** | throughput-bound, scheduler hides |
| `02_shmem.md` `bce8bf8` | **8.81×** (148×128, 10k iter) | multi-warp throughput |
| Q6_SMEM_TRANSPOSE | 8.2× | full transpose, single warp |

**Verdict**: D5's 5.74× is correct for single-warp latency-bound. V44/V45
split shows broader picture but contradicts itself with bce8bf8. Open
inconsistency listed in `02_shmem_CORRECTED.md` UNRESOLVED #1 and
`A_TO_D_RIGOR_AUDIT.md` UNRESOLVED #5.

---

## L1 capacity

| Source | Effective L1 | Pattern |
|---|---|---|
| D2 | **128 KB / 1024 lines** sharp | strided pointer-chase, 4 KB stride |
| V10 (per cache agent) | 2-4 KB smooth ramp | random Fisher-Yates chain |
| `03_caches_CORRECTED.md` | **Both correct under their access pattern** | reconciled |

**Verdict**: D2 + V10 are not in conflict. cache agent ratifies.

---

## L2 sector size

| Source | Sector | Sub-sector amp |
|---|---|---|
| D3 | **32 B** | 4B writes → 7× DRAM read amp |
| `03_caches_CORRECTED.md` §2.2 | 32 B | identical numbers |

**Verdict**: D3 ratified verbatim. No conflict.

---

## LOP3 properties

| Source | Latency | Throughput | RF ports |
|---|---|---|---|
| C3 | **4.5 cy** | 0.5/SMSP/cy = 14.16 TIPS @ 1500 | ≥3 (no port pressure) |
| V40 | (not separately) | 18.7 Glane/s = 48% FMA tier | (not separately) |
| `B300_TRUE_REFERENCE.md` | 4 cy lat, 2.0/SM/cy | matches | matches |

**Verdict**: C3 fully consistent with V40 (14.16 TIPS @ 1500 ⇒ 19.18 @ 2032
× 0.97 efficiency = ~18.7 Glane/s). RF-port finding doesn't conflict with
A4/D6 because LOP3 pipe rate (0.5/SMSP/cy) is below the RF 3-distinct
ceiling (0.66/SMSP/cy) — pipe is the bottleneck, not RF.

---

## Scoreboard depth

| Source | Claim |
|---|---|
| A3 | LDG scoreboard ≥32 in flight (no plateau through N=32) |
| Hopper whitepaper | 32-slot warp scoreboard per A2 |
| V40-V51 | not retested |

**Verdict**: A3 still HIGH for ≥32; MED for exact depth. No conflict; open.

---

## Open inconsistencies summary (unresolved across the audit)

1. **A6/B1 IADD3 = 0.50 vs V40 IADD3 = 0.67** — warp-count effect, unresolved
2. **A6 PRMT = 0.50 vs V40 PRMT = 0.36** — methodology, unresolved
3. **D5 5.7× vs V45 1× vs bce8bf8 8.81×** — bank conflict regime boundary, unresolved
4. **A3 scoreboard exact depth** — N=32 floor, real value open
5. **A6 SHFL+FFMA 14.7% mechanism** — single-warp explanation unverified
6. **A4/D6 reuse-cache vs operand-collector mechanism** — same observable, different physical model
7. **B2 LDG no-chain slower than chain** — queue backpressure hypothesis MED

---

## Notes on methodology

- A1's "empty-loop floor 23 cy" lesson (line 30) is METHODOLOGY GOLD —
  later misattributed in some catalog docs as "FFMA latency 23 cy"
  (legacy l.19565). Already retracted in `04_fp32_peak_CORRECTED.md` #3.
- A4's "self-op `fma a,a,a,a` is FAST not slow" finding (line 13)
  contradicts and supersedes legacy "8.46 cy self-op" claim already
  retracted in `04_fp32_peak_CORRECTED.md` #4.
- D6's `.reuse` SASS verification (255/256 vs 0/256) is the cleanest
  bit of evidence in the entire A-D series; should be cited whenever
  the RF port claim comes up.
