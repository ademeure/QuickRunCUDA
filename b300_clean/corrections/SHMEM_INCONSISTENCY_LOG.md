# SHMEM Bandwidth & Bank-Conflict Inconsistency Log

Cross-file audit, Topic = SMEM (excluding DSMEM, covered separately).
Sources audited:
- `02_shmem.md` (catalog)
- `V8_SMEM_BW.md`
- `D5_SMEM_BANK_BEHAVIOR.md`
- `Q6_SMEM_TRANSPOSE.md`
- `V41_V48_FINDINGS.md` (V44/V45)
- `B300_TRUE_REFERENCE.md`
- `CLAUDE.md` SMEM line
- user-memory `project_b300_v8_complete.md`, `project_b300_session4.md`

---

## A. Theoretical-peak: consistent

| File | Quoted | Notes |
|---|---|---|
| `02_shmem.md` | 38.49 TB/s @ 2032 (260 GB/s/SM) | rigorous derivation |
| `B300_TRUE_REFERENCE.md` | 38.5 spec | matches |
| `CLAUDE.md` | "Shared memory: 38.5 TB/s theoretical" | matches |
| `V8_SMEM_BW.md` | 36.4 @ 1920, 38.5 @ 2032 | both clocks shown |

No inconsistency.

---

## B. Headline measured peak

| File | Peak quoted | Recipe / clock |
|---|---:|---|
| `02_shmem.md` headline | **38.4 TB/s = 99.8%** | LDS.128, 1blk/SM, short run, 2032 boost (`d41c38c`) |
| `B300_TRUE_REFERENCE.md` | **38.4 TB/s = 99.8%** | matches `d41c38c` |
| `V8_SMEM_BW.md` | **26.9 TB/s = 74%** of 36.4 | plain `float` LDS, 8-ILP × 16 unroll, 1920 MHz |
| user memory `project_b300_session4.md` | "27.2 TB/s = 71%" | older session4, plain LDS |
| user memory `project_b300_v8_complete.md` | "26.9 TB/s = 74%" | matches V8_SMEM_BW |

**Inconsistency #1** (resolved by recipe): The 26.9 / 27.2 numbers are
**plain-scalar-LDS** ceilings, not SMEM-pipe SoL. The 38.4 number is
LDS.128 + RAW addr-chain + 1 blk/SM + 2032 boost. Both are correct for
their respective workloads. `02_shmem.md` already documents this in
"Findings being retired or corrected" §1.

**However**: `V8_SMEM_BW.md` ends with "Plain LDS at 74% is the practical
ceiling for scalar kernels on B300" and "To exceed ~27 TB/s, must use
matrix-load or async-load primitives" — this is **misleading** because
LDS.128 (a plain LDS variant, not ldmatrix) hits 99% with proper recipe.
V8's claim should add a forward pointer to `02_shmem.md §"Optimal recipe"`.

---

## C. ld.volatile.shared: confirmed RETRACTED

`02_shmem.md` lines 80, 243 explicitly retire the volatile-vs-non-volatile
distinction. SASS is identical, BW is identical (37.6 TB/s). All other
files in scope are silent on this; no live contradiction.

User-prompt mention: "Smem peak 17 → 35.9 TB/s (DCE fix via
`ld.volatile.shared`)" appears only in the user prompt; **no
`b300_clean/` file currently claims 17 TB/s SMEM peak with or without DCE
note**. The closest live "17 TB/s" mentions are all L2 (in `03_caches.md`),
not SMEM. The `02_shmem.md §Open Questions` 17–21 TB/s is the
**post-throttle sustained** number for 1920 MHz, not a DCE artifact, and
is correctly captioned. **No live retraction action needed in catalog.**

---

## D. Bank-conflict cost — REGIME SPLIT NOT IN CATALOG

| File | 32-way conflict cost | Test conditions |
|---|---:|---|
| `02_shmem.md` `bce8bf8` | **8.81× (multi-warp)** | 148 × 128 thr × 10k iter, 4R+1W/iter |
| `D5_SMEM_BANK_BEHAVIOR.md` | **5.74× (single-warp, dep chain)** | t=32 b=1, N_DEPS=8 |
| `Q6_SMEM_TRANSPOSE.md` | **8.2× (single-warp transpose)** | t=32, full 32×32 |
| `V41_V48_FINDINGS.md` V44 | **~2× (latency-bound chain)** | single chain |
| `V41_V48_FINDINGS.md` V45 | **~1× (throughput-bound, hidden)** | many warps |

**Inconsistency #2 (BIGGEST in this audit)**: The corpus offers FOUR
different headline cost numbers for "32-way bank conflict" depending on
which regime is tested (1×, 2×, 5.7×, 8.2×, 8.81×).

- The catalog `02_shmem.md` does NOT mention V44/V45.
- `V41_V48_FINDINGS.md` V44/V45 calls out the regime split but does not
  reconcile with the catalog's `bce8bf8` 8.81× number.
- `D5_SMEM_BANK_BEHAVIOR.md` §Caveats acknowledges single-warp
  pessimism but does not link to V45.

**Recommendation**: amend `02_shmem.md §Bank conflicts` to:
1. Note V44/V45 regime distinction.
2. Distinguish single-warp (D5/Q6/V44) vs multi-warp-contention (`bce8bf8`)
   vs many-warps-with-other-work-to-overlap (V45).
3. Drop "N-way conflict scales N/4" — only valid in one regime.

---

## E. SMEM atomic aggregate throughput — discrepant numbers

| Source | Number | Op |
|---|---:|---|
| `02_shmem.md §atomics` | 4.6 cy/op uncontended | INT32 atomicAdd |
| User memory `project_b300_v8_complete.md` `968e5b7` | **2.2 T-atomic/s aggregate** | INT atomicAdd, contention-invariant |
| User-prompt | "4.2 Tops/s no-contention" | unspecified op |
| User memory `project_b300_v8_complete.md` `2fc181b` | reduce-kernel uses SMEM atomic | not headline number |

**Inconsistency #3**: Catalog quotes per-op cycles only; user memory
quotes 2.2 T-atomic/s aggregate; user-prompt quotes 4.2 Tops/s. Without
a unified Tops/s in catalog, can't reconcile. **Suggest measurement** to
land a definitive aggregate number in `02_shmem.md`.

(Note: per `feedback_units_sanity` memory, atomic ops can be cache-line
combined and inflate 8× over byte BW; ensure aggregate "T-atomic/s"
specifies whether it's combined or per-instruction.)

---

## F. Capacity numbers — consistent

228 KB total / 227 KB opt-in / 1024 B reserved. All sources agree.
CLAUDE.md does not state SMEM capacity (no inconsistency, just gap).

---

## G. ldmatrix / stmatrix — consistent

`02_shmem.md` ldmatrix.x4 = 17.7 B/cy/warp, stmatrix.x4 = 14.2 B/cy/warp;
`B300_TRUE_REFERENCE.md` agrees on stmatrix W+R chain = 34.5 TB/s = 90%.
`V8_SMEM_BW.md` ldmatrix.x1 = 3.0 TB/s claim is consistent with x1's lower
per-instruction bytes (128 vs 512 for x4) — but the V8 conclusion
"ldmatrix doesn't improve SMEM BW" is **regime-specific** (x1, low ILP);
catalog (x4, proper recipe) gets 33–35 TB/s. **Already documented as a
recipe difference in 02_shmem; no live contradiction.**

---

## H. CLAUDE.md gap

CLAUDE.md only quotes the 38.5 TB/s theoretical, no measured peak. Suggest
amending to "Shared memory: 38.5 TB/s theoretical, 38.4 TB/s measured
(99.8%)" to bring the per-project memory in line with `02_shmem.md` and
`B300_TRUE_REFERENCE.md`.

---

## Summary of action items (suggested, not applied — originals unmodified)

1. **`02_shmem.md`**: add V44/V45 regime split to §Bank conflicts.
2. **`02_shmem.md`**: add aggregate Tops/s number for SMEM atomics.
3. **`V8_SMEM_BW.md`**: add forward pointer to `02_shmem.md` for the
   true LDS.128 SoL recipe; soften "27 TB/s practical ceiling" claim.
4. **`Q6_SMEM_TRANSPOSE.md`** and **`D5`**: add cross-link to V45
   throughput-regime caveat.
5. **`CLAUDE.md`**: add measured 38.4 TB/s alongside the 38.5 theoretical.
