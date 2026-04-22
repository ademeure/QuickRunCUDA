# B300 HBM3E DRAM Bandwidth — CORRECTED

**Audit basis:** cross-file diff of `01_hbm_bandwidth.md`, `B300_TRUE_REFERENCE.md`,
`V8_HBM_WRITE_SOL.md`, `V32_V40_FINDINGS.md`, `V41_V48_FINDINGS.md`,
`HBM_DATA_DEPENDENCE.md`, `L2_DRAM_DATA_PWR.md`, `CLAUDE.md` HBM section.
See `HBM_INCONSISTENCY_LOG.md` (this dir) for the full contradiction list.

**GPU:** NVIDIA B300 SXM6 AC, sm_103a, 148 SMs, 12 stacks HBM3E, 7680-bit
usable bus (post-ECC), 287.4 GB capacity. All measurements at 2032 MHz boost
unless noted.

---

## 0. Canonical theoretical peak (NORMALIZED)

**HBM3E spec, post-ECC: 7672 GB/s** = 7680 bit × 3996 MHz × 2 (DDR) / 8 / 1e9.

ALL "% of peak" numbers in this corrected file use **7672 GB/s** as the
denominator. Other documents in the corpus use:
- 7.31 TB/s (V32-V48 series) — empirical pure-direction peak, NOT spec.
- 8.0 / 8.18 TB/s (V8_HBM_WRITE_SOL.md, CLAUDE.md) — pre-ECC nominal,
  not achievable post-ECC.

When reading other docs, mentally re-normalize. See UNRESOLVED §A below.

---

## 1. Headline numbers (HIGH confidence — ncu DRAM bytes verified)

| Operation | Rate (TB/s) | % of 7672 spec | Recipe / source |
|---|---:|---:|---|
| **Read peak (canonical)** | **7.30** | **95.2%** | v8 + per-warp coalesced + non-persistent (`a04d9c8`, verified by `01_hbm_bandwidth.md` and `B300_TRUE_REFERENCE.md`) |
| Read peak (TMA bulk variant in 01) | 7.34 | 95.7% | TMA `cp.async.bulk` 8 KB chunks, 37888 blocks (commit cited in `01_hbm_bandwidth.md`) |
| Read peak (LDG.E.128 in 01) | 7.37 | 96.0% | LDG.E.128 with 37888 blocks (cited in `01_hbm_bandwidth.md`) |
| Read peak (A6 R-only) | 7.31 | 95.3% | A6 R:W=32:0 sweep (`01_hbm_bandwidth.md` R:W table) |
| **Write peak (canonical, NINJA recipe)** | **7.57** | **98.7%** ← SoL | NINJA recipe `e75c7e1` (1 v8 store/warp, max parallelism) — see §3 below. **Caveat: provenance is contested; see RETRACTIONS.** |
| Write peak (v8 + per-warp 32-iter) | 7.30 | 95.2% | a04d9c8 |
| **Concurrent R+W (50:50)** | **6.68** | 87.0% | minimum at any mix, A6 sweep |
| **Concurrent R+W (single-kernel best ratio)** | **7.31** | 95.3% | pure R or pure W (de3b4d5) |
| cudaMemset (true DRAM rate, ncu) | ~7.30 | ~95% | wall-clock 7.47-7.52 overstates by ~3% (RETRACTED elsewhere) |
| D2D copy (separate src/dst, NINJA) | 6.93 | 90.3% | `4958d6b` — beats cudaMemcpyAsync by 5.5% |
| D2D copy (`cudaMemcpyAsync`) | 6.56 | 85.5% | "single-direction 3.28 TB/s × 2" |

---

## 2. TMA path — PARTIALLY CORRECTED

V41-V48 found that pipelined TMA reads (8-deep) reach 7.20 TB/s, a 2.5×
improvement over V33's single-deep 6.72 TB/s. **However**, V41_V48
labelled this "NEW BEST = 98.5%" using 7.31 as denominator. Re-normalized
to the 7672 spec, V46 = 7.20 / 7672 = **93.8%**, which is **lower** than
01_hbm_bandwidth's quoted TMA bulk read (7.34 = 95.7%) and LDG.E.128
(7.37 = 96.0%).

**Corrected statement:** V46 confirms that pipelined TMA (8-deep) is
necessary to extract competitive TMA read bandwidth (vs single-deep), but
**does not establish a new architectural ceiling**. The current best
read SoL remains 7.30-7.37 TB/s = 95-96% of 7672 spec, attainable via
either LDG.E.128 + per-warp coalesced, or TMA with appropriate launch
geometry.

| TMA read variant | TB/s | % of 7672 | Notes |
|---|---:|---:|---|
| V33 single-deep 64 KB | 6.72 | 87.6% | confirmed via L2-cache fix |
| V46 pipelined 8-deep 16 KB | 7.20 | 93.8% | best in V41-V48 series |
| 01 TMA bulk 8 KB chunks | 7.34 | 95.7% | already higher than V46 |
| 01 LDG.E.128, 37888 blocks | 7.37 | 96.0% | the read SoL |

| TMA write variant | TB/s | % of 7672 | Notes |
|---|---:|---:|---|
| V34 TMA write 32 KB | 7.17 | 93.5% | (V41_V48 quoted 98% using 7.31 denom) |
| V47 TMA write 8-deep pipelined | 6.34 | 82.6% | "no benefit" — writes are already async fire-and-forget |

**TMA + prefetch.L2 anti-pattern (V42, NEW finding):** prefetch.L2 combined
with cp.async.bulk = **27% slower** than no-prefetch. TMA has its own DMA
path; explicit prefetch instructions block forward progress. Rule:
**never combine prefetch.L2 with cp.async.bulk**. (Note: the V6 1.58×
prefetch speedup was for legacy `cp.async`, NOT for `cp.async.bulk`;
no contradiction, but worth flagging.)

**TMA multicast aggregate (V32, V48):** 14.9 TB/s effective at 18 clusters
× 8-way multicast (V32). V48 attempted to pipeline multicast; capped at
13.96 TB/s — **multicast cannot be pipelined**, single engine per cluster.

---

## 3. Write peak — CONTESTED PROVENANCE

The "7.57 TB/s = 98.7%" headline from `B300_TRUE_REFERENCE.md` claims this
came from a STG-based NINJA recipe (commit `e75c7e1`, 1 v8 store per warp,
massive parallelism). However, `V8_HBM_WRITE_SOL.md` attributes 7.57 TB/s
to **TMA bulk store** (commit `28211ce`) and explicitly states that plain
STG.E.128 caps at 6.11 TB/s.

These two attributions are **mutually exclusive**:
- If TRUE_REFERENCE is right, plain STG with the right launch geometry hits 7.57.
- If V8 is right, plain STG caps at 6.11 and only TMA gets to 7.57.

Either way:
- The number 7.57 TB/s = 98.7% is real (both files agree on the value).
- One of the two attributions is wrong.

**RETAINED (HIGH conf):** 7.57 TB/s is the write SoL on B300.
**UNRESOLVED:** Which path produces it. Needs a clean re-test of both
e75c7e1 (NINJA STG) and 28211ce (TMA bulk) on the same machine with ncu
DRAM bytes verification. See UNRESOLVED §B.

V8_HBM_WRITE_SOL also claims TMA write "exceeds read peak by 5% (105%)";
this is a denominator-mismatch artifact (uses 7.2 effective for read, 8.0
nominal for write). Re-normalized: 7.57 / 7672 = 98.7% (still the SoL),
read peak 7.30-7.37 / 7672 = 95.2-96.0%, gap is ~3 pp not 10 pp. The
"reads slower than writes" framing is real but the "exceeds" framing
should be retired.

---

## 4. Optimal read recipe (HIGH confidence, unchanged from 01)

```cpp
__global__ void w_v8_coalesced(int *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *warp_base = data + warp_id * (32 * 1024 / 4);
    int v = 0xab;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = warp_base + (it * 32 + lane) * 8;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}
// Launch: <<<bytes / (256 * 1024), 256>>>   // 16384 blocks for 4 GB
```

SASS: `STG.E.ENL2.256` / `LDG.E.ENL2.256`.
Saturation requirements: WS ≥ 4 GB, 256-bit per-instruction width, per-warp
1-KB bursts, high parallelism. CTA-count and cache hint do not matter once
above are satisfied.

---

## 5. Working-set / cache effects (HIGH conf, unchanged from 01)

| WS | Effective BW | Tier |
|---|---:|---|
| 16 MB | 46.6 TB/s | L1 + L2 combined |
| 64 MB | 23.0 TB/s | L2 plateau |
| 100 MB | 20.9 TB/s | L2 edge |
| 126 MB | 8.2 TB/s | **CLIFF — exactly L2 capacity** |
| 256 MB | 7.32 TB/s | DRAM-bound |
| 1024 MB | 7.12 TB/s | DRAM-bound |
| 4 GB | 7.29-7.30 TB/s | DRAM-bound, true HBM3E ceiling |

Confirmed L2 capacity = `cudaDeviceProp.l2CacheSize = 126 MB` (132,644,864 B).

---

## 6. R:W ratio sweep (HIGH conf, unchanged from 01 A6)

| R:W (ops/thr) | DRAM read | DRAM write | Aggregate | % of 7672 |
|---|---:|---:|---:|---:|
| 32:0  | 7.31 | ~0   | **7.31** | 95.3% |
| 28:4  | 6.22 | 0.86 | 7.08 | 92.3% |
| 24:8  | 5.40 | 1.72 | 7.12 | 92.8% |
| 20:12 | 4.32 | 2.50 | 6.82 | 88.9% |
| 16:16 | 3.39 | 3.29 | **6.68** ← min | **87.0%** |
| 12:20 | 2.52 | 4.09 | 6.61 | 86.2% |
| 8:24  | 1.65 | 4.85 | 6.50 | 84.7% |
| 4:28  | 0.83 | 5.72 | 6.55 | 85.4% |
| 0:32  | ~0   | 7.28 | **7.28** | 94.9% |

U-shape, min at 50:50 = 6.68 TB/s. Confirms HBM3E is **shared bus, not
full-duplex**. Mechanism: tWTR/tRTW direction-switch penalty.

---

## 7. Data-dependence (HIGH conf via L2_DRAM_DATA_PWR.md)

L2-warm reads: 340.1 - 343.6 W across 11 patterns (3.5 W spread = 1%).
DRAM-cold reads: 522.6 - 528.0 W across 11 patterns (5.4 W spread = 1%).

**Memory subsystem is data-pattern INDEPENDENT** within 1%. In stark
contrast to tensor-core compute (20-35% variance). Mechanism: cache-line
traffic (fixed 128 B units), bus signaling at fixed rates, address
decoding deterministic per access.

`HBM_DATA_DEPENDENCE.md` gave a low-confidence inferred <50W answer with
a broken kernel (20.4 GB/s); superseded by L2_DRAM_DATA_PWR's clean 11-pattern
sweep. See RETRACTIONS.

---

## 8. D2D, multicast, and concurrent R+W (HIGH conf, unchanged)

- D2D `cudaMemcpyAsync` (2 GB): 3.28 TB/s single-direction = 6.56 TB/s R+W effective.
- D2D NINJA recipe: 6.93 TB/s R+W (HBM3E concurrent ceiling at separated stacks).
- TMA multicast aggregate: 14.9 TB/s effective (V32, 18 clusters × 8-way).
- TMA multicast cannot be pipelined (V48: 13.96 TB/s capped).

---

## RETRACTIONS

| Retracted claim | Source | Reason |
|---|---|---|
| "V46 = NEW BEST HBM read SoL 7.20 TB/s = 98.5%" | `V41_V48_FINDINGS.md` | Denominator was 7.31 (empirical), not 7672 (spec). Re-normalized: 93.8%. Below 01's 7.34 TMA bulk and 7.37 LDG. Demoted to "improves V33 single-deep but not architectural new SoL". |
| "TMA bulk store 7.57 TB/s = 105% (exceeds read peak)" | `V8_HBM_WRITE_SOL.md` | Denominator-mismatch artifact (effective 7.2 vs nominal 8.0). Re-normalized: 98.7%. Read-write asymmetry is real but ~3 pp not 10 pp. "Exceeds" framing retired. |
| "Plain STG.E.128 caps at 6.11 TB/s = 85% of HBM" | `V8_HBM_WRITE_SOL.md` | 01_hbm_bandwidth shows v8 STG with per-warp coalesced + non-persistent reaches 7.30 TB/s. The 6.11 ceiling is a launch-geometry artifact, NOT an architectural cap on STG. |
| "HBM3E spec = 8 TB/s" | `V8_HBM_WRITE_SOL.md`, `CLAUDE.md` | Pre-ECC nominal. Correct post-ECC spec is 7672 GB/s. |
| "HBM read peak 7.31 TB/s = theoretical" (used as denominator) | `V32_V40_FINDINGS.md` (V32-V36) | 7.31 is the empirical pure-direction peak, not the spec. Using as denominator inflates % numbers by ~5 pp. |
| "HBM data-dependence ≤50 W via memory streaming, throughput 20.4 GB/s" | `HBM_DATA_DEPENDENCE.md` | Test was broken (20.4 GB/s = 0.3% of peak). Superseded by `L2_DRAM_DATA_PWR.md` (11 patterns, <1% variance). |
| "TMA multicast can be pipelined for further speedup" | implicit assumption | V48 disproved: capped at 13.96 TB/s vs V32's 14.9 TB/s. Multicast = single engine per cluster. |
| All retractions previously listed in `01_hbm_bandwidth.md` § "Findings being RETIRED" | (preserved) | Still retired; included by reference. Includes "97-98% memset", "DMA fast path", "10.4 TB/s duplex bonus", "9% read>write asymmetry", "5.16 TB/s ld.global", "8.17 TB/s pre-ECC", "21% memset>user". |

---

## UNRESOLVED

### A. Which is the canonical HBM3E theoretical peak?

The corpus uses three values:
1. **7672 GB/s post-ECC** — derivation in 01_hbm_bandwidth.md (7680 bit × 3996 MHz × 2 / 8 / 1e9). This is what NVIDIA quotes for the actual usable bandwidth post-ECC overhead.
2. **8183.8 GB/s pre-ECC** — 8192-bit × 3996 × 2 / 8. Includes the 1024-bit ECC reservation. NOT usable.
3. **8 TB/s nominal** — datasheet rounded number; may include or exclude ECC depending on context.

**Recommendation: ANCHOR on 7672 GB/s.** All `% of peak` columns in this
corrected doc use this denominator. CLAUDE.md should be updated to cite
7672 GB/s, not "~8 TB/s spec".

The CLAUDE.md value 7.31 (mentioned in the audit task) appears to actually
be in section 1 of CLAUDE.md as "HBM3E: ~7-7.5 TB/s read peak (matches 8 TB/s spec)"
— it does not say "7.31 theoretical". The 7.31 in V32-V48 is the empirical
pure-direction peak, used as denominator by convention. Both are
internally consistent if you accept the convention, but they make
"% of peak" numbers across files NON-COMPARABLE.

### B. Provenance of the 7.57 TB/s write SoL

- `B300_TRUE_REFERENCE.md` says it came from NINJA recipe `e75c7e1` (STG-based, 1 v8 store per warp).
- `V8_HBM_WRITE_SOL.md` says it came from TMA bulk store `28211ce`.

These cannot both be the source. **Action needed:** rerun both commits
on the same machine with ncu DRAM bytes verification. Likely outcome:
- Plain NINJA STG hits 7.57 (TRUE_REFERENCE correct, V8 used inferior STG recipe).
- OR TMA hits 7.57 and best STG caps lower (V8 correct, TRUE_REFERENCE attribution wrong).

### C. Why V46's 7.20 TMA pipelined < 01's 7.34 TMA bulk?

01 quotes 7.344 TB/s for TMA bulk read with 37888 blocks. V46 quotes 7.20
TB/s with pipelined 8-deep. Either:
- 01's 7.344 was already implicitly pipelined via launch geometry (37888
  blocks = 256 blocks/SM provides natural concurrency).
- OR 01's 7.344 is over-stated.
- OR V46's 7.20 leaves something on the table that 01 captures.

Reproducing both back-to-back with identical ncu metrics would settle it.

### D. The 5% spec gap (7.30 vs 7672)

01 lists this as open: refresh, command bus, row-precharge candidates.
A2 partially addressed (bursts <1 KB hit 98.6%, longer bursts under-saturate).
Still no direct attribution of the budget.

### E. Multi-GPU contention on shared HBM (open from 01)

2× B300 in the same chassis, both saturating HBM. Untested. Multi-GPU
notes in MEMORY.md exist but address NVLink P2P, not HBM-side contention.

### F. Refresh-rate sensitivity (open from 01)

No measurement of long-sustained streaming hitting a refresh-induced floor.

### G. Stride / channel-locality sweep (open from 01)

Does going outside contiguous 1 KB bursts (e.g. 4 KB or 256 B per warp)
preserve 7.30 TB/s, or is the 95% number tied tightly to per-warp 1 KB?

### H. cudaMemset wall-clock vs ncu gap (open from 01)

The wall-clock 7.5 TB/s of cudaMemset hints at a marginally faster pattern
nobody has reproduced from user PTX. SASS extraction of cudaMemset's driver
kernel would settle it (and `B300_TRUE_REFERENCE.md` notes 6 hook methods
all blocked).

---

## Files of record

Same as `01_hbm_bandwidth.md`, plus:
- `b300_clean/V32_V40_FINDINGS.md` (TMA SoL series, A6 R:W denomination quirk)
- `b300_clean/V41_V48_FINDINGS.md` (V46 pipelined TMA, V42 prefetch antipattern)
- `b300_clean/V8_HBM_WRITE_SOL.md` (contested 7.57 TB/s attribution)
- `b300_clean/L2_DRAM_DATA_PWR.md` (authoritative data-dependence)
- `b300_clean/B300_TRUE_REFERENCE.md` (master summary; see contested 7.57 attribution)
- `b300_clean/HBM_DATA_DEPENDENCE.md` (SUPERSEDED by L2_DRAM_DATA_PWR.md)
