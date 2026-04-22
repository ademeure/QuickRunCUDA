# L1 / L2 / TMEM Inconsistency Log

Cross-document inconsistencies found in the b300_clean catalog, organized
by topic. For each, the **resolution** is the value that should be cited
going forward; deltas are the magnitude of the disagreement.

---

## L2 BANDWIDTH (the big one)

The catalog has L2 BW numbers spanning 10–36 TB/s. They are not all wrong;
they measure different things. Inconsistency comes from *not labelling* the
metric.

| # | Doc | Line | Claim | Metric implied | Status |
|---|---|---|---|---|---|
| 1 | `CLAUDE.md` | line 86 | "L2 bandwidth: 10–36 TB/s reported" | union of all | Honest range; should add metric breakdown |
| 2 | `B300_TRUE_REFERENCE.md` | 31 | **23.85 TB/s** kernel-effective | SM-side w/ L1 reuse | HIGH, current canonical |
| 3 | `B300_TRUE_REFERENCE.md` | 32 | **13.30 TB/s** lts bus traffic | wire | HIGH, same kernel as #2 |
| 4 | `M5_MEMORY_CHEATSHEET.md` | 25 | "L2 single-SM 23 TB/s" | aggregate-equivalent | matches #2 |
| 5 | `03_caches.md` | §3a | 17 TB/s `.cg` carveout=100, 8–128 MB | wire-ish, modern | HIGH |
| 6 | `03_caches.md` | §3b | 22–26 TB/s `.cg` carveout=0 | mixed | MED, not re-verified |
| 7 | `03_caches.md` | §3c | 30–36 TB/s `.ca` WS<2 MB | LSU/L1 dispatch ceiling | mis-labelled as L2 |
| 8 | `V8_L2_BW_VERIFIED.md` | line 14 | 13.85 TB/s `.cg` 64 MB | wire | HIGH, matches #3 |
| 9 | `V8_L2_BW_VERIFIED.md` | line 1 | "L2 BW = 13.85 TB/s" headline | wire | HIGH |
| 10 | `V41_V48_FINDINGS.md` | line 90 | "V33 L2-cache (10.84→6.72)" | retraction example | HIGH (rule 3 catch) |
| 11 | `POPCOUNT_WRITES.md` | line 81 | 10.84 TB/s `lts__t_bytes` | wire under writes | HIGH |

**Resolution:** Always label L2 BW with one of three metrics:
- **kernel-effective** (SM-delivered, L1 amplification on): ~23.85 TB/s
- **wire / lts** (pure L2 partition output): ~13.30 TB/s
- **L1-amplified small-WS peak** (NOT L2; LSU dispatch): ~30 TB/s

The 1.79× ratio between kernel-effective and wire is the L1 hit rate.

---

## L2 CAPACITY

| # | Doc | Claim | Status |
|---|---|---|---|
| 1 | `B300_TRUE_REFERENCE.md`, `03_caches.md`, `M5_MEMORY_CHEATSHEET.md` | **126 MB** (132,644,864 B) | HIGH, canonical |
| 2 | older catalog | 256 MB | RETRACTED — unit confusion |
| 3 | older catalog | 50 MB | RETRACTED — single-side meas |
| 4 | older catalog | 280 / 192 / 186 MB | RETRACTED — scope confusion |
| 5 | `L2_BITSTRIDE_SWEEP.md` and `L2_POPCOUNT_SWEEP.md` | "fits in 96 MB L2" | **MINOR ERROR** — should say 126 MB. The 96 MB value is wrong but is used only as upper-bound on a 64 MB WS, so doesn't change the conclusion. |

---

## L1 CAPACITY

| # | Doc | Claim | Status |
|---|---|---|---|
| 1 | `03_caches.md` | unified pool **256 KB**, L1 portion 20–228 KB | HIGH |
| 2 | `D2_L1_CAPACITY_RIGOR.md` | "Per-SM L1 ≈ 128 KB = 1024 lines" sharp boundary | HIGH (default carveout) |
| 3 | `V10_L1_CAPACITY.md` | "effective L1 ~2–4 KB random access" | HIGH (different access pattern) |
| 4 | `M5_MEMORY_CHEATSHEET.md` | "L1/SMEM 256 KB" total + "128 KB L1 effective" | HIGH |
| 5 | older catalog | "L1 = 32 KB" | RESOLVED — at default (max-SHMEM) carveout |

**Resolution:** "L1 size" depends on (carveout, access pattern). All numbers
above are correct in their context. Always state carveout AND access pattern.

---

## L1 LATENCY

| Doc | Latency | Clock | Notes |
|---|---|---|---|
| `D2_L1_CAPACITY_RIGOR.md` | 39 cy chained | 1500 MHz | strided pointer-chase |
| `03_caches.md` §11 | 42–45 cy | 2032 MHz | warm pointer-chase |
| `V10_L1_CAPACITY.md` | 47.5 cy at 1 KB | unknown | random Fisher-Yates |
| `M5_MEMORY_CHEATSHEET.md` | 38 cy | unknown | catalog summary |

Spread 38–47 cy across clock × pattern. **No conflict** — within ±20% noise.

---

## L1 BANDWIDTH

| Doc | BW | Notes |
|---|---|---|
| `V8_L2_BW_VERIFIED.md` | 30.5 TB/s | strided default-ld, ws fits in L1 |
| `M5_MEMORY_CHEATSHEET.md` | ~46 TB/s | older / optimistic / different ILP |

**Status:** moderate disagreement (1.5×). Likely different unrolling / per-SM
aggregation. Flag M5's 46 TB/s as **MED** until re-derived. Use **30.5 TB/s**
as the conservative measured peak, with 46 TB/s as the ILP-maxed upper bound.

---

## L2 LATENCY

| Doc | Value | Notes |
|---|---|---|
| `03_caches.md` §11 | 300–310 cy avg, near 310 / far 660 | @ 1920 MHz |
| `M5_MEMORY_CHEATSHEET.md` | 152 ns / 230 cy | likely per-side near |
| `B300_TRUE_REFERENCE.md` line 86 | 164 ns local atomic L2 round-trip | round-trip not single-side hit |

**Status:** consistent if you separate near/far. 230–310 cy is the near-side
range; 660 cy is far-side. The 1.27–2.4× near-vs-far ratio is universally
agreed. M5's "230 cy" is plausible for the near-side common case.

---

## L2 SECTOR / LINE SIZE

Universally agreed across `D3_L2_SECTOR_RIGOR.md`, `M5_MEMORY_CHEATSHEET.md`,
`03_caches.md`: **128 B line, 32 B sector (4 sectors/line)**. No conflict.
Sub-sector 4 B writes incur 7× DRAM read amplification; well-replicated in D3.

---

## L2 ATOMIC UNIT COUNT

| Doc | Claim | Notes |
|---|---|---|
| `B300_TRUE_REFERENCE.md` line 162 | "~32 L2 atomic units" | catalog |
| `L2_UNITS_REFINED.md` | ~32.5 (27 / 0.83) | derivation from two test scenarios |
| `CLOCK_DOMAINS_AND_L2_UNITS.md` | ~27 active in parallel = 84% of catalog 32 | direct measurement |

**Status:** catalog "~32" is the architectural unit count; "~27" is what
actually saturates per video cycle (84% utilization). Both consistent.

---

## TMEM CAPACITY

| Doc | Value | Status |
|---|---|---|
| `06_tensor_cores.md` §76 | **256 KB/CTA** = 512 cols × 128 lanes × 4 B | HIGH |
| `M5_MEMORY_CHEATSHEET.md` line 13 | "TMEM 256 KB / 38 MB total" | HIGH (38 MB = 148 × 256 KB chip aggregate) |
| `CURIOSITY_LIST_V4.md` D7 | "TMEM capacity 256 KB" | HIGH |

No conflict. Worth clarifying: 256 KB is per-CTA; 38 MB chip-wide assumes
1 CTA per SM at full occupancy.

---

## TMEM BANDWIDTH

| Doc | Read | Write | Status |
|---|---|---|---|
| `06_tensor_cores.md` §76 | **~60 TB/s** chip | 97–131 TB/s | HIGH (current canonical) |
| `CURIOSITY_LIST_V4.md` D7 | 65 TB/s (57 B/cy/warp) | — | HIGH |
| older CATALOG line 1265-1293 | 830 TB/s | — | RETRACTED — DCE-inflated |
| older CATALOG | 295 TB/s | — | RETRACTED — DCE-inflated |

**Resolution:** **~60 TB/s** read, ~97–131 TB/s write. Never cite 830 / 295 TB/s.

---

## V33 RULE-3 CASE STUDY (canonical example for the swarm)

`V32_V40_FINDINGS.md` line 20 documents the V33 catch:
> "Initial 10.84 TB/s (148% of theoretical) → rule 3 violation → caught
> L2 cache reuse"

Final V33 number: **6.72 TB/s** (= 92% of HBM theoretical), then improved
in V46 to **7.20 TB/s** (98.5%) via 8-deep TMA pipelining.

The 10.84 TB/s figure is real but it is the **L2 wire BW** for that test
configuration (matches `L2_BITSTRIDE_SWEEP.md` line 81 `lts__t_bytes`,
matches `POPCOUNT_WRITES.md` line 81). It was mistakenly *labelled* as
HBM BW because the test author didn't notice the working set was being
served from L2 not DRAM.

**Lesson for any future HBM-BW test on B300:**
- If WS < 126 MB or any unroll-with-modulo pattern, the per-pass touched
  bytes may fit in L2.
- ncu `dram__bytes` is the only way to confirm DRAM is actually being hit.
- Any HBM BW number > 7.7 TB/s (theoretical spec) is impossible — instant
  rule-3 fail.

---

## "L2 = 22 TB/s" (CLAUDE.md line ?? — may not exist verbatim)

The investigation prompt says CLAUDE.md memory section claims "L2 = 22 TB/s".
The actual line in `CLAUDE.md` (line 86) is:
> "**L2 bandwidth: depends heavily on access pattern; ranges 10-36 TB/s reported**"

The "22 TB/s" specific value isn't in CLAUDE.md but it sits in the middle
of the range and matches the carveout=0 catalog MED number from
`03_caches.md` §3b. Treat "22 TB/s" as a **rule-of-thumb mid-range** that
straddles the 13.30 wire and 23.85 kernel-effective metrics.

---

## SUMMARY OF ACTIONS

1. **Always label L2 BW** with one of {kernel-effective, wire/lts, L1-amplified}.
2. **Always state carveout** when quoting L1 size.
3. **Always state access pattern** (strided vs random) when quoting L1 effective capacity.
4. **Always confirm with ncu `dram__bytes`** before claiming HBM peak.
5. **Update `L2_BITSTRIDE_SWEEP.md` and `L2_POPCOUNT_SWEEP.md`** to say
   "fits in 126 MB L2" not "96 MB L2" (cosmetic — does not change conclusion).
6. **Re-verify** `M5_MEMORY_CHEATSHEET.md` "L1 ~46 TB/s" against V8's 30.5 TB/s.
7. **Stop citing** 830 / 295 TB/s TMEM, 50 MB / 256 MB L2, "L1 = 32 KB" without carveout.
