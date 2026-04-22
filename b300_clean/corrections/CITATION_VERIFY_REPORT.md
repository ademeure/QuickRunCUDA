# Citation integrity report — `B300_CANONICAL_REFERENCE.md`

Run date: 2026-04-22
Subject: `/root/github/QuickRunCUDA/b300_clean/B300_CANONICAL_REFERENCE.md` (17 631 lines)
Confidence-tagged claims scanned: **130** (`[🟢|🟡|🔴|⚫ ...]` blocks)
Unique `src:` citation strings: **69**
Unique file paths extracted across all `src:` blocks: **117**

Doc convention (line 257 of canonical): _"`src:` paths are relative to `b300_clean/`"_. So bare filenames are intended to resolve to `b300_clean/<file>`; entries that begin with `corrections/` resolve to `b300_clean/corrections/<file>`.

## Verdict

**Citation integrity: SOLID with minor cosmetic gaps.**

- 0 broken-in-spirit citations (no claim is unsupported)
- 1 typo'd filename (`HEADLINE_v5.md` → should be `HEADLINE_CORRECTIONS_v5.md`)
- 2 path-prefix issues (`.cu` files cited bare-name but live outside `b300_clean/`)
- ~15 bare-filename citations that should arguably be `corrections/<name>` for unambiguity, but the file IS uniquely resolvable so this is a stylistic nit, not a defect
- 3 user-memory pointers (`project_b300_*.md`) that are intentional cross-system refs, not file-tree citations

---

## Resolution summary

| Category | Count | Notes |
|---|---:|---|
| Resolved at cited path | 85 | Exact hit at `b300_clean/<cited>` or `<cited>` (utils/CLAUDE) |
| Resolved at corrections/ subdir (bare cite ambiguous) | 15 | Cite says `FOO_CORRECTED.md`; file lives at `corrections/FOO_CORRECTED.md`. No collision elsewhere, so unambiguous in practice. |
| Path prefix `b300_clean/` redundantly used | 12 | E.g., `src: b300_clean/02_shmem.md` works because both `b300_clean/02_shmem.md` exists and the convention prepends `b300_clean/`. Slightly inconsistent style. |
| WRONG-PATH (file exists elsewhere) | 2 | `cluster_raw_barrier.cu`, `cluster_sass_audit.cu` actually live in `/root/github/QuickRunCUDA/investigations/` |
| MISSING from repo | 3 | `HEADLINE_v5.md` (typo), `project_b300_multigpu.md`, `project_b300_v6_complete.md` (these latter two exist in user memory at `~/.claude/projects/.../memory/` and the doc text self-identifies them as "user memory") |

---

## MISSING citations (with line numbers)

| Cited path | Canonical line | Likely fix |
|---|---:|---|
| `HEADLINE_v5.md` | 13282 | **TYPO** — should be `corrections/HEADLINE_CORRECTIONS_v5.md` (used 29× elsewhere). Single occurrence. |
| `project_b300_multigpu.md` | 1186, 2222 | NOT a defect — file exists at `~/.claude/projects/-root-github-QuickRunCUDA/memory/project_b300_multigpu.md`. The doc text already disambiguates ("User memory `project_b300_multigpu.md`"). Could prefix with `~/.claude/.../memory/` for explicitness. |
| `project_b300_v6_complete.md` | 480, 1302, 2473 | Same as above — exists in user memory; cited as "user memory" inline. Not a broken reference. |

## WRONG-PATH citations

| Cited path | Canonical line | Real location |
|---|---:|---|
| `cluster_raw_barrier.cu` | 4636, 5349 | `/root/github/QuickRunCUDA/investigations/cluster_raw_barrier.cu` |
| `cluster_sass_audit.cu` | 5349 | `/root/github/QuickRunCUDA/investigations/cluster_sass_audit.cu` |

Both live OUTSIDE `b300_clean/`, so the relative-to-b300_clean convention can't reach them. Suggested fix: cite as `../investigations/cluster_raw_barrier.cu` or include the absolute prefix once in §1.

---

## Top-20 most-cited files

| Rank | File | Cite count |
|---:|---|---:|
| 1 | `corrections/01_hbm_bandwidth_CORRECTED.md` | 6 |
| 1 | `CLAUDE.md` | 6 |
| 3 | `V41_V48_FINDINGS.md` | 5 |
| 3 | `B300_TRUE_REFERENCE.md` | 5 |
| 5 | `V52_RUN_RESULTS.md` (resolves to `corrections/V52_RUN_RESULTS.md`) | 4 |
| 5 | `16_power_clock_CORRECTED.md` (resolves to `corrections/`) | 4 |
| 5 | `08_sync_primitives.md` | 4 |
| 8 | `corrections/TCGEN05_DEDUP_CONSOLIDATED.md` | 3 |
| 8 | `corrections/06_tensor_cores_CORRECTED.md` | 3 |
| 8 | `V9_THREADFENCE_COST.md` | 3 |
| 8 | `POWER_INCONSISTENCY_LOG.md` (resolves to `corrections/`) | 3 |
| 8 | `DSMEM_REFERENCE.md` | 3 |
| 13 | `corrections/HEADLINE_CORRECTIONS_v5.md` | 2 |
| 13 | `corrections/HBM_STACKS_INDEPENDENT_VERIFY.md` | 2 |
| 13 | `corrections/HBM_DENOMINATOR_FINAL.md` | 2 |
| 13 | `corrections/CONFIDENCE_LADDER.md` | 2 |
| 13 | `corrections/07_atomics_CORRECTED.md` | 2 |
| 13 | `corrections/03_caches_CORRECTED.md` | 2 |
| 13 | `V9_OP_LATENCY.md` | 2 |
| 13 | `V10_DVS_CURVE.md` | 2 |

All top-20 files exist on disk and are reachable via the documented path convention.

Note on legacy (non-`corrections/`) sources: 4 of the top 12 (`V41_V48_FINDINGS`, `B300_TRUE_REFERENCE`, `V9_THREADFENCE_COST`, `08_sync_primitives`, `DSMEM_REFERENCE`) are original wave-1 files surviving uncontradicted into Wave 7. Per Appendix D.1.2 those are explicitly listed as "still valid, no corrections needed" — citation use is consistent with the supersession map.

---

## Quick-nav vs body headline consistency (10-row spot check)

The doc has 6 quick-nav rows; expanded with 4 additional headline numbers from the at-a-glance card.

| Quick-nav claim | Body section | Body Answer line | Match? |
|---|---|---|---|
| FFMA 74.62 TFLOPS at 2032 boost (96.92% of 76.96) | §16 line 2729 | "FFMA peak = 74.62 TFLOPS = 96.92% of 76.96 theoretical at 2032" | YES |
| HBM read peak 7.30 TB/s = 95.2% of 7680 | §6 line 860 | "7.30–7.37 TB/s = 95.2–96.0% of 7680 GB/s" | YES |
| L2 capacity 126.5 MB; 13.30 wire / 23.85 effective | §11 line 1495 | "126.5 MB ... 13.30 TB/s wire ... 23.85 TB/s kernel-effective" | YES |
| Dual-issue: pipe_alu 98% + pipe_fma 49% = 147% | §22 line 3494 | "pipe_alu = 98.0% AND pipe_fma = 49.4%" | YES |
| FP8 e4m3 cuBLAS 3984 TFLOPS | §25 line 4160 | "4425 TFLOPS zero-data / 3984 TFLOPS random data realistic" | YES |
| NVFP4 cuBLAS 11423 TFLOPS = 76.2% of 15 PF | §46 line 9726 | "NVFP4 cuBLAS+cudaGraph BPG=16 reaches 11423 TFLOPS (76.2% of 15 PF spec)" | YES |
| FP32 theoretical 76.96 TFLOPS | §16 | matches | YES |
| 2× B300 NVLink P2P 0.778 TB/s = 86% | §14 line 2222 | "P2P read 0.778 TB/s = 86% of 900 GB/s/dir spec" | YES |
| 8 HBM stacks (NOT 12) | Appendix D + §6 | "8 stacks of 12-Hi" + HBM_STACKS_INDEPENDENT_VERIFY | YES |
| Boost 2032 / locked 1920 | §3 line 480 | "Boost clock is 2032 MHz; nvidia-smi -lgc 2032 paradoxically pins to 1920" | YES |

**0 inconsistencies** found between quick-nav and body answers.

---

## Appendix D provenance map spot check (10-row)

D.1.1 claims every original `b300_clean/N_*.md` has a `corrections/N_*_CORRECTED.md` counterpart.

| Original | Corrections counterpart | Both exist? |
|---|---|---|
| `01_hbm_bandwidth.md` | `corrections/01_hbm_bandwidth_CORRECTED.md` | YES |
| `02_shmem.md` | `corrections/02_shmem_CORRECTED.md` | YES |
| `04_fp32_peak.md` | `corrections/04_fp32_peak_CORRECTED.md` | YES |
| `06_tensor_cores.md` | `corrections/06_tensor_cores_CORRECTED.md` | YES |
| `08_sync_primitives.md` | `corrections/08_sync_primitives_CORRECTED.md` | YES |
| `10_launch_overhead.md` | `corrections/10_launch_overhead_CORRECTED.md` | YES |
| `12_nvlink_p2p.md` | `corrections/12_nvlink_p2p_CORRECTED.md` | YES |
| `14_math_intrinsics.md` | `corrections/14_math_intrinsics_CORRECTED.md` | YES |
| `16_power_clock.md` | `corrections/16_power_clock_CORRECTED.md` | YES |
| `DSMEM_REFERENCE.md` | `corrections/DSMEM_CORRECTED.md` | YES |

Auxiliary/spawned files Appendix D references (also spot-checked):
`META_DOCS_CORRECTED.md`, `M_SYNTHESIS_CORRECTIONS.md`, `NVFP4_CONSOLIDATED.md`, `V8_V10_MISC_CORRECTED.md`, `TCGEN05_POWER_CONSOLIDATED.md`, `MATH_INCONSISTENCY_LOG.md`, `SASS_VERIFY_DUAL_ISSUE.md`, `V9_GRAPH_LAUNCH.md` — **all 8 exist**.

---

## Section-anchor (`§N`) spot check

The doc cites `§N` inside `corrections/X` files several times. Verified the referenced sections exist:

| Citation | Target file | Section present? |
|---|---|---|
| `01_hbm_bandwidth_CORRECTED.md §0` | line 14 of file | YES |
| `01_hbm_bandwidth_CORRECTED.md §1+§2` | lines 28, 46 | YES |
| `01_hbm_bandwidth_CORRECTED.md §3` | line 87 | YES |
| `01_hbm_bandwidth_CORRECTED.md §6` | line 158 | YES |
| `01_hbm_bandwidth_CORRECTED.md §7` | line 177 | YES |
| `01_hbm_bandwidth_CORRECTED.md §8` | line 193 | YES |
| `03_caches_CORRECTED.md §1` | line 21 | YES |
| `03_caches_CORRECTED.md §2` | line 79 | YES |
| `06_tensor_cores_CORRECTED.md §R4` | line 82 | YES |
| `STRAYS_CORRECTED.md §2` | line 46 | YES |

**0 wrong-section** issues found.

---

## Recommendations

1. **Fix the typo at line 13282**: change `HEADLINE_v5.md` to `corrections/HEADLINE_CORRECTIONS_v5.md`. (1 char change in spirit; the only confirmed defect.)
2. **Optional: prefix `corrections/` on the 15 bare-filename CORRECTED cites** so the `src:` is unambiguous without relying on the convention preface. E.g., `META_LESSONS.md` → `corrections/META_LESSONS.md`. Currently they all resolve uniquely so this is cosmetic.
3. **Optional: fix the 2 `cluster_*.cu` cites** to `../investigations/cluster_raw_barrier.cu` (or move the convention to allow off-`b300_clean/` paths explicitly).
4. **Optional: prefix user-memory cites** with `~/.claude/projects/-root-github-QuickRunCUDA/memory/` for the 3 `project_*` references. Currently disambiguated by inline text only.

No load-bearing claim in the canonical doc is missing its source.
