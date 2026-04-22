# 02 — Shared Memory CORRECTED

Cross-file reconciliation of SMEM bandwidth, bank-conflict, atomic, and capacity claims.
Sources: `02_shmem.md`, `V8_SMEM_BW.md`, `D5_SMEM_BANK_BEHAVIOR.md`,
`Q6_SMEM_TRANSPOSE.md`, `V41_V48_FINDINGS.md`, `B300_TRUE_REFERENCE.md`,
`CLAUDE.md`, plus user memory `project_b300_v8_complete.md` /
`project_b300_session4.md`.

System: B300 SXM6, sm_103a, 148 SMs.
Theoretical SMEM peak: **38.49 TB/s @ 2032 MHz boost** (32 banks × 4 B × 148 SMs × 2.032 GHz).
At 1920 MHz "locked": 36.4 TB/s.

---

## 1. Verified bandwidth per access pattern

| Pattern | BW (TB/s) | %peak (vs 38.5) | Clock | Confidence | Source / commit |
|---|---:|---:|---|---|---|
| **Pure LDS.128 read, RAW addr-chain, 1blk/SM, short run** | **38.4** | **99.8%** | 2032 boost | HIGH | `d41c38c` (`rigor_smem_sol.cu`); SASS+ncu verified |
| LDS.128, 4 SMSPs | 38.0 | 99% | 2032 | HIGH | `ninja_smsp_vec.cu` |
| LDS.128, 2 SMSPs | 35.4 | 92% | 2032 | HIGH | same |
| `ld.shared.v4.u32` non-volatile | 37.6 | 98% | 2032 | HIGH | `d41c38c` (volatile==non-volatile, identical SASS) |
| float4 typical | 35–36 | 92% | 2032 | HIGH | 02_shmem.md |
| ldmatrix.x4.b16 (tensor feed) | 33–35 | 91% | 2032 | HIGH | `4ccda4f`, `664a67b` |
| stmatrix W+R chain | 34.5 | 90% | 2032 | HIGH | `8bd85e8` (TRUE_REFERENCE) |
| Read+write mix (4R+1W/iter) | 27.2 | 71% | 2032 | HIGH | `4503a17` |
| Plain `float` LDS, 8-ILP × 16 unroll | 26.9 | 74% (of 36.4 @1920) | 1920 | HIGH | V8_SMEM_BW.md, `352ab1f` |
| 8 × scalar LDS.32 | 19–26 | 50–67% | 2032 | HIGH | `4503a17` |
| Sustained (>8000 iter, post-throttle) | 17–21 | ~50% (of 36.4) | 1920 throttled | MED | 07_smem_peak.md §6 |

**Headline SoL: 38.4 TB/s = 99.8%** (`02_shmem.md` and `B300_TRUE_REFERENCE.md` agree).
**Realistic mixed-workload ceiling: 27.2 TB/s** for read+write tile work.

---

## 2. Bank-conflict regime: latency vs throughput (V44/V45 update)

The catalog `02_shmem.md` and `Q6_SMEM_TRANSPOSE.md` both report bank-conflict
slowdowns from **single-warp / latency-bound** measurements. V41–V48's V44+V45
finding shows this is **only half the story**:

| Regime | 32-way conflict cost | Source |
|---|---:|---|
| Latency-bound (single warp, dependent chain) | ~2× (V44 = chain-serial; D5 = 5.74×; Q6 = 8.2× full transpose) | V44, D5, Q6 |
| Throughput-bound (many warps, scheduler hides serialization) | **~1× (effectively free)** | V45 |
| 02_shmem.md "banks_proper" multi-warp | **8.81×** (148×128, 10k iter) | `bce8bf8` |

**Inconsistency**: The catalog's `bce8bf8` 32-way = 8.81× slowdown is from a
multi-warp throughput test, NOT a latency test. This contradicts V45's
"~1× hidden" claim under the same nominal regime. The discrepancy is
**unresolved** — likely the V45 setup had enough other warps queued
to hide the conflict, while `bce8bf8` was contention-saturated.

CUDA C Programming Guide implies serialization model (~32×). Real-world
B300 cost is **regime-dependent** between 1× and 8.8×; the naive 32× is
never observed.

`02_shmem.md` does NOT carry the V44/V45 nuance.
`D5_SMEM_BANK_BEHAVIOR.md` is single-warp only and acknowledges this in §Caveats.
`Q6_SMEM_TRANSPOSE.md` is single-warp full-transpose; its 8.2× is a real
workload number, but conflict-cost generalization should NOT use it.

---

## 3. SMEM atomics

| Op / contention | Cost | Source |
|---|---:|---|
| INT32 atomicAdd uncontended | 4.6 cy | 02_shmem §atomics, `baeef1f` |
| INT32 atomicAdd 32-way | 4.6 cy (zero penalty) | same |
| FP32 atomicAdd uncontended | 85 cy | same |
| FP32 atomicAdd 32-way | 5729 cy (67×) | same |
| Aggregate INT atomic peak (all SMs, all-lanes-same-addr) | **~2.2 Tatomic/s** | user memory `project_b300_v8_complete.md` (`968e5b7`) |

**4.2 Tops/s no-contention**: not reproduced in catalog; user memory says
**2.2 T atomic/s aggregate**. The "4.2" figure may have been mis-recalled
or come from a different op (atomicInc/Dec which are 4 ns vs add 8 ns =
~2× faster — could account for the ~2× discrepancy). MED.

**Recommendation**: catalog quotes 4.6 cy/op uncontended INT atomic;
~2 T-atomic/s aggregate. Use INT atomics for SMEM histograms, not FP32.

---

## 4. Capacity (HIGH, device-attribute verified)

| Limit | Value |
|---|---:|
| Total SRAM per SM (L1+SHMEM unified) | **256 KB** |
| `cudaDevAttrMaxSharedMemoryPerBlockOptin` | **228 KB** (227 KB usable + 1 KB reserved) |
| Reserved SMEM/block | 1024 B |
| Chip-wide SRAM | 148 × 256 = 37.9 MB |

`B300_TRUE_REFERENCE.md` says "SHMEM/SM = 228 KB total"; `02_shmem.md` says
227 KB opt-in/block + 1 KB reserved (= 228 KB). **Consistent.**

CLAUDE.md does NOT state SMEM capacity; only quotes 38.5 TB/s peak.

---

## RETRACTIONS

1. **"SMEM peak ≈ 17 TB/s plain LDS"** — User-memory line says
   `Smem peak 17 → 35.9 TB/s (DCE fix via ld.volatile.shared)`. **No file in
   `b300_clean/` currently quotes 17 TB/s as a SMEM peak.** The 17 TB/s figure
   in `03_caches.md` refers to **L2 BW** (carveout=100), not SMEM.
   The `17–21 TB/s sustained` line in `02_shmem.md §Open Questions` is the
   throttled-clock long-run regime, NOT a "DCE" artifact and is correctly
   captioned. The "ld.volatile.shared" claim itself is RETRACTED in `02_shmem.md`:
   volatile and non-volatile emit identical `LDS.128` and deliver identical
   37.6–38.4 TB/s. **No retraction needed in any catalog file** — the
   pre-DCE-fix 17 TB/s claim does not appear there. (The user-memory line is
   the artifact and may be stale.)

2. **"DSMEM 0.8% slower than local"** — RETRACTED (LICM); true ratio 7–8×
   latency, 9× throughput. `02_shmem.md` already documents.

3. **"DSMEM 4.7× slower"** — RETRACTED (FADD-serialized accumulator turned
   throughput test into latency test). `02_shmem.md` already documents.

4. **"DSMEM 37.3 TB/s ≈ peak"** (user memory `71934d0`) — User memory
   `project_b300_v8_complete.md` already DOWNGRADES this to "ALL DSMEM
   numbers REJECTED — SASS shows LD.E (global-style) not LDS, ncu wavefront
   count 0.0001% of expected = DCE'd". The `B300_TRUE_REFERENCE.md` value
   of 3.06 TB/s aggregate is the correct one.

5. **"ld.volatile.shared.v4.u32 unlocks 1.8× more BW"** (B300_PIPE_CATALOG §0)
   — RETRACTED in `02_shmem.md`. Same SASS, same BW.

6. **`ldmatrix_test.cu` (commit `d714801`)** — DCE'd, RETRACTED.

7. **`stmatrix DCE` (commit `6077386`)** — RETRACTED; replaced by post-fix
   `ede88fb` numbers (32–36 cy/warp).

---

## UNRESOLVED

1. **V44/V45 throughput-vs-latency split for bank conflicts is not
   reflected in `02_shmem.md` or `D5_SMEM_BANK_BEHAVIOR.md`.** The
   catalog's multi-warp `bce8bf8` test reports 8.81× for 32-way, but V45
   says throughput-bound 32-way is ~1×. Need a single test that varies
   warp count to map the regime boundary.

2. **SMEM atomic aggregate throughput discrepancy: catalog cites cycles only;
   user memory says 2.2 T-atomic/s (`968e5b7`); user prompt says 4.2 T-ops/s.**
   No single in-tree file gives an aggregate Tops/s number. Needs measurement.

3. **Sustained vs burst BW transition (17–21 TB/s after ~8000 iters at
   1920 MHz) is unexplained beyond clock — power management implicated
   but no signal pinpointed.** (02_shmem.md §Open Questions.)

4. **L1/SHMEM partition non-monotonic** in 8–40 KB band (commit `879f942`).
   MED-confidence; needs predictive model from `cudaFuncSetAttribute` settings.

5. **DSMEM bank-conflict atomic scaling** — the 32-way INT atomic test
   reported zero penalty, but bank-aware scaling for atomics has not been
   mapped (02_shmem.md §Open Questions).

6. **CLAUDE.md SMEM line gives only the 38.5 TB/s theoretical** — no
   measured peak; mildly inconsistent with the corpus headline of 38.4
   TB/s measured. Suggest amending CLAUDE.md to add "(measured 38.4 = 99.8%)".
