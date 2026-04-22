# NVLink / PCIe / Multi-GPU Inconsistency Log

Cross-doc audit by NVLink/PCIe sub-agent. Originals untouched.
Companion files: `12_nvlink_p2p_CORRECTED.md`, `13_pcie_system_CORRECTED.md`.

---

## Severity legend

- **CRIT**: factual error that propagated to TRUE_REFERENCE or memory.
- **HIGH**: internal contradiction within or across the catalog.
- **MED**: number drift within ±15% across docs.
- **LOW**: wording or formula nit; numbers OK.

---

## Inconsistency table

| # | Sev | Loc(s) | Issue | Resolution |
|---|---|---|---|---|
| 1 | CRIT | 13_pcie line 6, 216; CLAUDE memory | "NVLink v7" — wrong generation | Use **NVLink 5** (Blackwell B300). NVLink 4 = Hopper, NVLink 5 = Blackwell. There is no "NVLink 7". |
| 2 | CRIT | TRUE_REFERENCE line 36-37 | Uses "spec 757" (NVLink-4 H100 marketing) as denominator for B300 NVLink-5 measurements → claims "1.04× spec" | Use 900 GB/s/dir spec. Recompute: 778 / 900 = 86%, 720 / 900 = 80%. |
| 3 | CRIT | TRUE_REFERENCE line 38 | "PCIe 0.058 = 23% of 256 spec — CPU-bound" | Root cause is UNCONFIRMED per 13_pcie §"Hypotheses (none verified)". Strike "CPU-bound" or annotate as hypothesis. |
| 4 | HIGH | 12_nvlink §1 vs §12 | §1 says "NV18 = 18 NVLink5 links per direction"; §12 says "NV18 = 18 NVLink5 links" (total). | NV18 = 18 links total (each full-duplex). Strike "per direction" in §1. |
| 5 | HIGH | 12_nvlink §3a vs §3c vs §11 vs M5 vs TRUE_REF | Read BW reported as 778 / 820 / 860 / 740 / 780 in 5 places. | Canonical: payload **778 GB/s**, ncu link RX **860 GB/s** (includes protocol). 820 in §3c summary is unsourced. |
| 6 | HIGH | 12_nvlink §3b vs §3d vs §11 vs M5 vs TRUE_REF | Write BW reported as 710 / 718 / 720 / 765 / 836. | Canonical: payload **720 GB/s**, ncu link TX **836 GB/s**. M5 "740" averages read+write — drop. |
| 7 | HIGH | 12_nvlink §12 retirement table | Says "NVLink BW 757 / 1503: CONFIRMED" — but body uses 900/1800 spec. Self-contradicts. | Rewrite as "NVLink-5 BW 900 / 1800 spec; measured 778 / 1543 = 86%". |
| 8 | HIGH | 13_pcie §"4 copy engines" line 17, 71 | "1.72× sum of either alone" — math gives 0.86×. Wording bug. | Should read "1.72× single-direction (98.8 / 57.5)". |
| 9 | MED | M5_MEMORY_CHEATSHEET line 28 | "NVLink (peer) 740 GB/s, 77% of 956" — neither read nor write number; basis unclear. | Replace with explicit read=778 / write=720 entries. 956 = "raw" link cap including protocol; 900 is data spec. |
| 10 | MED | M5 line 28 | Uses 956 as denominator; TRUE_REFERENCE uses 757; 12_nvlink uses 900. **Three different denominators across docs.** | Standardize on 900 GB/s/dir (= 18 × 50). Note 956 raw and 757 NVLink-4 are both wrong here. |
| 11 | MED | 12_nvlink §11 ncu vs §3 measured | NVLink TX 836 vs payload 720 = 14% gap; NVLink RX 860 vs payload 778 = 11%. Larger than 6% protocol overhead would predict. | Likely includes ECC + headers per flit; could also be ncu over-counting retried flits. Annotate. |
| 12 | MED | 12_nvlink §4a "1.55 µs at 1920 MHz" vs TRUE_REFERENCE coordination ladder line 87 "1662 ns" | 1.55 µs at 1920 MHz vs 1.66 µs in TRUE_REF. Reasonable agreement (±7%). | Acceptable; clock mode ambiguity noted. |
| 13 | LOW | 13_pcie line 50 | "Gen 5 effective ... 100 GB/s full-duplex" but actual measurement is 98.8 GB/s. | Round-up wording, not a real error. |
| 14 | LOW | 13_pcie line 41 | "Gen 6 x16 theoretical: ~256 GB/s/direction". PCI-SIG official is closer to 242 GB/s after FEC. | Acceptable rough number. |
| 15 | LOW | 12_nvlink §5a "REMOTE contend > unique" surprise | Explanation given (warp coalescing reduces packet count). Plausible but not independently re-tested. | Mark as MED, could use confirmation. |

---

## Reconciliation: are 718 (memory), 720 (catalog write), 740 (M5), 765 (sweep saturation) the same number?

Project memory says: "718 GB/s write, 820 GB/s read, 49 Gatomic/s LOCAL all-contend, 16 Gatomic/s REMOTE."

Comparing:

| Source | Write GB/s | Method |
|---|---:|---|
| Memory | 718 | MGFenceBench |
| TRUE_REF | 710 | 4096 blocks v8 |
| 12_nvlink §3b W=128 event | 720 | "MGFenceBench cross-checked" |
| 12_nvlink §3b W=1024 steady | 768 | event-timed |
| 12_nvlink §3d 74 SMs | 765 | sweep saturation |
| 12_nvlink §11 ncu TX | 836 | nvlink__data_transmitted |
| M5 cheatsheet | 740 | "J2 (#e16901f)" — likely an avg |

The 710/718/720 cluster IS the same measurement, ±2%. The 765-768 are
"steady state with deeper warp width". The 836 is link-level (protocol
+ payload). The 740 in M5 is suspicious — might be a (read+write)/2
average that doesn't make physical sense (NVLink is full-duplex).

**Conclusion**: Memory's 718 + 820 are **consistent with** the 12_nvlink
catalog's 720 + 778 (within measurement noise). The cross-doc drift is
accounting choice (event vs ncu vs averaged), not real disagreement.

---

## Reconciliation: NVLink generation claim sources

| Claim | Source | Verdict |
|---|---|---|
| "NVLink 5" | 12_nvlink_p2p.md, TRUE_REFERENCE.md | **CORRECT** |
| "NVLink v7" | 13_pcie_system.md, CLAUDE.md memory | **WRONG** — fix |
| "(no version)" | M5_MEMORY_CHEATSHEET.md | underspecified |

NVIDIA's marketing for Blackwell consistently says "5th generation NVLink"
or "NVLink 5". The "v7" in this catalog likely came from confusing a
nvidia-smi field or a misread of "PCIe Gen 6 + NVLink" → "NVLink v7".

---

## Recommended single-source-of-truth update

Apply these to TRUE_REFERENCE.md when consolidating:

```
| **NVLink-5 P2P read (payload)** | **0.778** | 86% of 900 GB/s/dir spec | 12_nvlink §3a |
| **NVLink-5 P2P read (link RX)** | **0.860** | 96% of spec — includes protocol | ncu |
| **NVLink-5 P2P write (payload)** | **0.720** | 80% of spec | 12_nvlink §3b |
| **NVLink-5 P2P write (link TX)** | **0.836** | 93% of spec | ncu |
| **NVLink-5 P2P bidi aggregate** | **1.543** | 86% of 1800 GB/s/dir | sym |
| **PCIe Gen 6 x16 H2D pinned** | **0.0577** | 90% of Gen 5 / 23% of Gen 6; root cause unconfirmed | 13_pcie |
| **PCIe Gen 6 x16 D2H pinned** | **0.0574** | 90% of Gen 5 | 13_pcie |
| **PCIe full-duplex aggregate** | **0.0988** | 1.72× single-dir | 13_pcie |
```

And purge "NVLink v7" / "spec 757" / "CPU-bound" claims.
