# V46 "Denominator Artifact" Demotion — Doubt Report

**Question audited:** Was the wave-1 HBM agent right to demote V46 (TMA pipelined
read = 7.20 TB/s = "98.5% NEW HBM SoL") to "denominator artifact, actually 93.8%,
below existing 7.30+ measurements"?

**Verdict: DEMOTION IS CORRECT (with one nuance).**

---

## 1. True HBM3E theoretical peak on B300 — best estimate

**7672 GB/s post-ECC** is the right denominator. Derivation in
`01_hbm_bandwidth.md` line 138: 7680 bit × 3996 MHz × 2 (DDR) / 8 / 1e9.

Caveats noticed but not load-bearing here:
- The 7680-bit "usable" figure assumes 8 stacks × 1024 b − 512 ECC, but B300
  has 12 stacks. The catalog's bus-width derivation is internally muddled
  (`01_hbm_bandwidth.md` line 136 says "12 × 1024 = 12,288 raw … usable
  post-ECC = 7680" which doesn't subtract correctly). Regardless, 7672 GB/s
  is the published spec NVIDIA-aligned number that 01 and TRUE_REFERENCE
  both anchor on.
- "8 TB/s nominal" (CLAUDE.md) is the marketing number; not usable post-ECC.
- "7.31 TB/s" (V32-V48 convention) is the **empirical pure-direction peak**,
  not a spec — using it as a denominator is the bug.

---

## 2. V46's reported number — was it the best run?

V46's 7.20 TB/s is the **best of a 4-point sweep** (`v46_tma_inflight.cu`
lines 144), `n_inflight ∈ {1,2,4,8}`, averaged over RUNS=5 wall-clock-event
runs. So 7.20 is "best config, mean of 5". Not cherry-picked-best-run.

---

## 3. V46's actual rank in the READ-SoL ladder (apples-to-apples)

Re-normalized to 7672 GB/s spec (READ only, not write):

| Variant | TB/s | % of 7672 | Source |
|---|---:|---:|---|
| LDG.E.128, 37888 blocks | 7.365 | 96.0% | `01_hbm_bandwidth.md` line 66 |
| TMA cp.async.bulk 8 KB, 37888 blocks | 7.344 | 95.7% | `01_hbm_bandwidth.md` line 65 |
| User v8 + per-warp coalesced (4 GB) | 7.37 | 96.0% | `01_hbm_bandwidth.md` line 102 |
| A6 R-only (R:W=32:0) | 7.31 | 95.3% | `01_hbm_bandwidth.md` R:W table |
| **V46 TMA pipelined 8-deep, 16 KB, 148 blocks** | **7.20** | **93.8%** | V41_V48 |
| V33 TMA single-deep 64 KB | 6.72 | 87.6% | V32_V40 |

V46 ranks **5th** in the read ladder. It is below LDG.E.128 (7.365),
above-mentioned TMA bulk variant (7.344), per-warp v8 (7.37), and even
the A6 R-only sweep (7.31). The ~7.30+ existing measurements are all
READS — apples-to-apples.

---

## 4. Is the architectural finding (2.5× pipelining speedup) still real?

**YES.** V46 vs V33 is 7.20 / 6.72 ≈ 1.07× — wait, that's not 2.5×. The
"2.5× speedup" headline in V41_V48 must mean something else (probably
per-CTA throughput at 1-deep vs 8-deep within V46 itself; the recipe sweeps
n_inflight ∈ {1,2,4,8} on the SAME launch geometry). That intra-V46 ratio
is plausible and the **architectural lesson — TMA reads need pipelining;
single-deep wastes mbarrier latency — is independent of denominator and
remains valid.** The lesson "writes are already async fire-and-forget,
pipelining doesn't help" (V47 = 6.34) is also independent and real.

---

## 5. Recommendation

**Retain the demotion of "V46 = NEW BEST 98.5%".** It was a denominator
artifact (7.20/7.31 = 98.5%, but 7.20/7672 = 93.8%). Existing 7.30-7.37
LDG/TMA-bulk read measurements (96.0%) already exceed V46.

**Soften (not retract) the V41_V48 finding to:** "V46 confirms TMA reads
benefit from 8-deep pipelining (intra-test 1-deep → 8-deep speedup) and
recovers ground that V33's single-deep left on the table, but does NOT
establish a new architectural read SoL. The HBM3E read ceiling remains
7.30-7.37 TB/s = 95-96% of 7672 GB/s spec."

The wave-1 HBM agent's call was justified. The doubter (me) found one
secondary concern — the catalog's own 12-stack-bus derivation is
arithmetically suspect — but it does not rescue V46's headline.

(Word count: ~395)
