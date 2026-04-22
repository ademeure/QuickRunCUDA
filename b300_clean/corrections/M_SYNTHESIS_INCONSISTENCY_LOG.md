# M-Synthesis Cross-File Inconsistency Log

Cross-document inconsistencies among M1 - M16 themselves AND between the M-files
and newer/peer references (V32-V51, B300_TRUE_REFERENCE, peer category corrections).
Originals NOT modified.

Format per row: topic - which docs disagree - resolution.

---

## 1. PIPE OVERLAP (FFMA + ALU dual-issue)

| Source | Claim | Notes |
|--------|-------|-------|
| M1 B1 | FFMA + IADD3 = 14.2% overlap | Different denominator |
| M8 row | FFMA + IADD3 = 56% | Same-warp |
| M10 H1 | (cites M8 matrix) | Inherits |
| V40 | FFMA / IADD3 both ~26 Glane/s on FMA pipe | Same pipe |
| V49 (501134a) | FFMA + LOP3 = 55% overlap factor | Same-warp |
| V49 (501134a) | FFMA + IADD3 = 54% overlap factor | Same-warp |
| V50 (fbe1c18) | Warp-specialized = 74% | New variant |

**Resolution:** M1's 14.2% and M8's 56% use different denominators but tell the same
story: same-warp FFMA + ALU is partial. V49 establishes the canonical 54-55%
"overlap factor of perfect parallelism". V50 adds warp-specialized 74%. M8 must
add a warp-spec column; M16 myth-bust on "114 TOPS" needs rephrasing.

---

## 2. MUFU + FFMA "free" claim

| Source | Claim |
|--------|-------|
| M1 B3 | FFMA + MUFU ~100% overlap up to 4 FFMA/MUFU |
| M7 architectural summary | "FFMA + MUFU should be ~90%+" |
| M8 row | MUFU + FFMA = 100%+ super-linear |
| V49 | dual-issue is partial; issue port shared |
| V41 | MUFU at 1/(4cy)/SMSP - dispatched once per 4 cycles |

**Resolution:** "100%+ super-linear" is the result of MUFU's 1/(4cy) issue rate
fitting into FFMA's bubbles - NOT genuine dual-pipe parallelism. The framing should
be "FFMA fills MUFU's dispatch gaps", not "MUFU is free behind FFMA".

---

## 3. MUFU peak rate

| Source | Claim | Regime |
|--------|-------|--------|
| M14 row | rsqrt 47.8 GMUFU/s = 99.49% XU | 1-chain self-dep |
| M16 table I | XU peak 47.8 GMUFU/s @ 99.5% | (cites M14) |
| V41 (1b7e168) | MUFU non-EX2 = 4.74 Gops/s/chip | Saturated |
| V41 | EX2 = 9.22 Gops/s = 2; faster | Saturated outlier |
| 14_math sec 1 | MUFU peak ~4.8 TGOps/s | Confirms V41 |
| MATH_INCONSISTENCY_LOG #3 | M16 XU row mislabeled | Resolution |

**Resolution:** 47.8 G is single-chain rsqrt latency-bound; saturated MUFU pipe peak
is 4.74 G/chip. M14 / M16 rows are mislabeled by 100; (read as a saturated peak).
Already flagged by math agent.

---

## 4. L1 bandwidth

| Source | L1 BW |
|--------|-------|
| M5 line 25 | ~46 TB/s (catalog) |
| M14 line 36 | 30.5 TB/s = ~100% |
| CACHES_INCONSISTENCY_LOG #L1 BW | 30.5 TB/s conservative; 46 TB/s ILP-max upper |

**Resolution:** Per CACHES log: M14 = strided default-ld measured peak = HIGH; M5's
46 TB/s = older / optimistic / different ILP = MED. Cite 30.5 TB/s; reserve 46 TB/s
as ILP-max upper bound.

---

## 5. HBM read SoL

| Source | Peak |
|--------|------|
| M5 line 26 | 7.30 TB/s (95% of 7672 spec) |
| M14 line 41 | 5.82 TB/s = 81% (LDG coalesced) |
| V46 (d332321) | 7.20 TB/s = 98.5% (TMA 8-deep pipelined) |
| V9 cp.async row | 6.91 TB/s = 96% (cp.async.ca) |
| B300_TRUE_REFERENCE | 7.30 TB/s read; 7.57 TB/s NINJA write |

**Resolution:** Multiple regimes. Cite 7.20 TB/s (V46) as the TMA-pipelined peak
or 7.30 TB/s (per B300_TRUE_REFERENCE) as the HBM SoL. M14's 5.82 TB/s is plain
LDG only - update with TMA-pipelined and NINJA rows.

---

## 6. HBM write SoL

| Source | Peak |
|--------|------|
| M5 (no explicit write) | -- |
| M14 line 42 | 6.11 TB/s = 85% (STG.E.128) |
| M14 line 43 | 7.57 TB/s = 95% (TMA bulk store) |
| B300_TRUE_REFERENCE | NINJA recipe 7.57 TB/s = 98.7% spec |

**Resolution:** Both 6.11 (plain STG) and 7.57 (TMA / NINJA) are accurate at their
regime. B300_TRUE_REFERENCE upgrades NINJA to "98.7% spec" framing.

---

## 7. Idle / static power

| Source | Idle |
|--------|------|
| M2 | 164.7 W true idle (1500 MHz implied) |
| M11 | 165-170 W "regardless of utilization" |
| M13 (no explicit idle) | -- |
| POWER_INCONSISTENCY_LOG #A | 144 W @ 510, 167 W @ 1500, 197 W @ boost |

**Resolution:** M2 is correct AT 1500 MHz; M11's "regardless of utilization" is the
inconsistency - idle scales 144-198 W with clock. Add clock context to all M-file
idle quotes.

---

## 8. FFMA TFLOPS / W

| Source | Energy |
|--------|--------|
| M11 | FFMA-bound 9.0 J/TFLOP = 0.111 TF/W (at 359 W / 39.7 TFLOPS) |
| 16_power_clock (POWER #J) | 0.21 TF/W (74.6 TF / 361 W) |
| M14 (catalog "137 GFLOPS/W" mention) | 0.137 TF/W at 1500 MHz lock |

**Resolution:** M11's 39.7 TFLOPS is HALF-PEAK; B300_TRUE_REFERENCE FFMA peak is
74.62 TF (96.92%). M11 measured a low-occupancy regime mis-labeled as "FFMA-bound".
Use 0.21 TF/W (16_power_clock) for FFMA peak; 0.137 TF/W at 1500 MHz lock.

---

## 9. HMMA peak

| Source | TF |
|--------|-----|
| M14 | 578.6 TFLOPS (99.90%) |
| M7 | (no explicit) |
| B300_TRUE_REFERENCE row 47 | 569 TF (matches catalog "burst 569") |
| B300_TRUE_REFERENCE row 58 | "single-chain 1543 was over-counted; RETRACTED" |

**Resolution:** 578.6 vs 569 is a small spread (1.7%); both within measurement
noise of full peak. Cite 569 (B300_TRUE_REFERENCE canonical).

---

## 10. Stuck-at-1005 silent failure

| Source | Mention |
|--------|---------|
| User memory feedback_clock_stuck_no_lock | EXPLICIT |
| M2 | NOT mentioned |
| M9 | NOT mentioned |
| M14 | NOT mentioned (uses boost 2032 throughout) |
| B300_TRUE_REFERENCE | Implicit at line 378 (recovery via -rgc) |

**Resolution:** Add stuck-at-1005 caveat to all M-files using boost-clock numbers.
Currently undocumented in M-files; can silently invalidate measurements.

---

## 11. Cluster MAX

| Source | Claim |
|--------|-------|
| M3_TOPOLOGY_CHEATSHEET | "8 verified, 16 per spec" |
| V5 C5 | "Cluster MAX = 8 (CSIZE=16 silently fails)" |
| B300_TRUE_REFERENCE row 137 | "Max usable = 16 (non-portable), 8 portable" |
| M10 | Cluster MAX = 8 |

**Resolution:** All consistent; "16" is non-portable opt-in. Practical = 8.

---

## 12. cluster.barrier latency

| Source | Cy |
|--------|----|
| M3_TOPOLOGY F4 | 395 cy floor |
| M7 §1 sync hierarchy | cluster.barrier 395 |
| V5 C2 | 390 cy O(1) up to 8 |
| M15 / M16 ladder | 370 cy / 182 ns |

**Resolution:** 370-395 cy is within noise (±7%). Cite 390 cy (V5 C2 canonical).

---

## 13. Pipe ownership: where does IADD3 live?

| Source | Pipe |
|--------|------|
| M8 | "ALU (IADD3)" - separate from FMA |
| M16 table I | ALU = IADD3, LOP3.LUT, SEL, ISETP |
| V40 (d1d09c5) | IADD3 = FMA pipe (26 Glane/s, top tier) |
| V41/V49 | LOP3/IMUL/PRMT = INT-bit / permute pipes (half rate) |

**Resolution:** V40 is the most rigorous; IADD3 lives on FMA pipe. M8/M16 grouping
"ALU = IADD3 + LOP3 + ..." is incorrect - IADD3 should move to FMA group; the rest
are INT-bit / permute / compare pipes.

---

## 14. Per-FFMA energy at 1500 MHz

| Source | pJ/FFMA |
|--------|---------|
| M2 | 4.4 pJ/FFMA (with .reuse, broadcast) |
| M11 | 4.4 pJ baseline |
| V5 D2 | 4.4 pJ/FFMA at 1500 MHz |
| V6 D4 | DVS scaling: 3.1 (510) -> 9.96 (1920) pJ/FFMA |

**Resolution:** All consistent at 1500 MHz. Add DVS row for clock-sweep context.

---

## 15. cudaGraph launch / ExecUpdate

| Source | Claim |
|--------|-------|
| M3 launch overhead | cudaGraph launch = 512 ns (4; faster than direct) |
| M10 H6 | Build 1.0 vs 0.85 us; ExecUpdate 4-16; faster |
| M12 | (refers M10) |
| B300_TRUE_REFERENCE row 105 | ExecUpdate 1.4 us = 25; faster than reinstantiate |
| M16 myth-bust #4 | "cudaGraph reduces per-kernel launch latency" -> FALSE for single-kernel |

**Resolution:** All consistent if read as "graph batching" not "single launch".
M16's myth-bust is correct; cudaGraph wins at N=100+ kernels (3.84;) per M16.

---

## 16. Persistent kernel power

| Source | Claim |
|--------|-------|
| M13 #10 | 148 SMs / 256 thr SPIN: +7.4 W; NANOSLEEP: +2.8 W (62% savings) |
| V5 R2 | mbarrier.try_wait saves 20 W vs spin (-10.4%) |
| M2 idle table | mbarrier.try_wait 171.6 vs spin 173.1 W |
| V7 J4 | "persistent kernel adds ~0 W" (small block count) |

**Resolution:** Persistent kernel power scales with active blocks. 1-block ;0 W,
148-block +7-20 W with 60% savings via mbarrier/nanosleep over spin. M13's framing
is more accurate at full-occupancy.

---

## 17. NVLink P2P BW

| Source | Peak |
|--------|------|
| M3_TOPOLOGY J2 | 740 GB/s = 77% of 956 |
| M13 #7 | 778 GB/s = 86% of NVLink v7 ~900 spec |
| M14 row | 778 GB/s = 86% |
| B300_TRUE_REFERENCE | 778 GB/s read; 711 GB/s write |

**Resolution:** 740 (J2) and 778 (V8) are within 5%; cite 778 GB/s (V8 is more
recent + 4096 blocks recipe). NVLink theoretical is 757 (per CLAUDE.md memory) or
~900 (per M13/M14). Use 757 as conservative.

---

## 18. Cross-file: "no SM power-gating" claim

| Source | Claim |
|--------|-------|
| M11 | "Static GPU 165-170 W regardless of utilization" |
| M12 #5 | "NO SM power-gating (static constant)" |
| V7 K4 | NO SM power-gating verified |
| POWER #A | Idle scales 144-198 W with CLOCK (DVFS, not power-gating) |

**Resolution:** Two distinct mechanisms - SM power-gating (none) vs DVFS (scales
with clock). M11/M12 conflate these. Correct framing: "no SM-level power-gating;
overall idle DOES scale with clock state via DVFS".

---

## Cross-cutting summary

| M-file | Major issues | Conf |
|--------|--------------|------|
| M1 | Reframe B1/B3 with V49 numbers | LOW issue |
| M2 | Add clock context to idle | LOW |
| M3 | Both M3 docs OK | None |
| M4 | OK | None |
| M5 | L1 = 30.5 TB/s (V8); TMA = LDG row outdated | MED |
| M6 | OK (single-block primitives stand) | None |
| M7 | "Two FFMA sub-pipes" misleading wording | LOW |
| M8 | **CRITICAL**: pipe attribution + missing warp-spec column | HIGH |
| M9 | OK; matches user memory | None |
| M10 | Inherits M8 issues | LOW |
| M11 | **CRITICAL**: half-peak labeled "FFMA-bound"; idle wrong | MED |
| M12 | "Boost no throttle" needs precision qualifier | LOW |
| M13 | OK | None |
| M14 | **CRITICAL**: MUFU peak mislabeled; HBM peaks superseded by V46 | HIGH |
| M15 | Latency ladder solid; mbarrier 123 cy provenance unclear | LOW |
| M16 | **CRITICAL**: XU peak mislabeled; ALU pipe attribution wrong | HIGH |

## Recommended supersession order

1. **M14 / M16** are highest-priority for retraction-overlay because they are most
   cited and contain mis-labeled peaks (MUFU 47.8 G; HBM 5.82 / 6.11 TB/s).
2. **M8** needs warp-spec column from V50; pipe attribution fix from V40.
3. **M11** needs half-peak vs peak disambiguation.
4. M-files cited in user-facing reports should add header pointing to this log.

## Items NOT in CLAUDE.md but in M-files needing canonicalization

- M9's "ML inference USE BOOST CLOCK 3;" matches user memory - canonical.
- M13's "Queue depth = 1024 per stream" - canonical, USER-CORRECTED.
- M14's NINJA SoL recipes - canonical via B300_TRUE_REFERENCE forward-reference.
- M16's 5 myth-busts - canonical (rigor-verified).
