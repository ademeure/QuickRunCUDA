# 13 — PCIe + System — CORRECTIONS

Source-of-truth review of `b300_clean/13_pcie_system.md` against
`B300_TRUE_REFERENCE.md`, `M5_MEMORY_CHEATSHEET.md`, project memory,
and CLAUDE.md.

Originals untouched.

---

## RETRACTIONS

### R1. "NVLink v7" naming (line 6, line 216)
- 13_pcie_system.md says "NVLink v7 (NV18, 18 lanes)" and
  "GPU0 ↔ GPU1 = NV18 (NVLink v7, 18 lanes)".
- **Wrong.** B300 uses **NVLink 5th generation**, not "NVLink 7".
- See `12_nvlink_p2p_CORRECTED.md` R1 for the full naming-drift table.
- Also "18 lanes" is informal; nvidia-smi calls them "links". Each link
  has multiple SerDes lanes internally; a link is the connection unit.
- **Correction**: replace both occurrences with **"NVLink 5 (NV18 = 18
  links)"**.

### R2. "PCIe Gen 6 x16 H2D 0.058 TB/s = 23% of 256 spec — CPU-bound" (TRUE_REFERENCE line 38)
- The "CPU-bound" attribution is **NOT verified**. 13_pcie_system §"PCIe
  Gen 5 vs Gen 6" lists THREE possible causes, none confirmed:
  1. Host slot/root complex/retimer negotiates Gen 6 PHY but data path
     runs Gen 5 (BIOS/SBIOS config).
  2. AMD EPYC 9575F IOD doesn't deliver Gen 6 DMA rates.
  3. PLX switch / re-driver in chassis is Gen 5 only.
- The doc itself flags these as "Hypotheses (none verified — needs
  Gen 6 host)".
- **Correction**: TRUE_REFERENCE should not assert "CPU-bound". It
  should say "0.058 TB/s = 23% of Gen 6 spec / 91% of Gen 5 spec; root
  cause UNCONFIRMED — PHY negotiates Gen 6 but data path runs at
  Gen 5 effective rate".

### R3. "256 GB/s spec" denominator drift
- 13_pcie_system §"Theoretical accounting" line 223:
  "PCIe Gen 6 x16: 64 GT/s × 16 lanes × FLIT/PAM4 ≈ 256 GB/s/direction".
- TRUE_REFERENCE uses 256 spec.
- M5_MEMORY_CHEATSHEET line 29: "PCIe Gen 6 ×16 | 58 GB/s effective"
  with no spec column — internally consistent but doesn't show the
  spec gap.
- The 256 number is approximately correct (PCIe 6 actual is 242 GB/s
  unidir per PCI-SIG marketing after FEC; "256" is a rounded number).
  Acceptable but worth a footnote.
- **Correction**: state explicitly "Gen 6 x16 spec = 242-256 GB/s/dir
  depending on accounting; we use 256 as round number".

### R4. Inconsistency between M5 cheatsheet "740 GB/s NVLink" and 13_pcie body
- M5 line 28: "NVLink (peer) | 740 GB/s | 77% of 956".
- 13_pcie does not contradict but DOES NOT include this number for
  cross-reference. The 740 number reconciles to neither 12_nvlink_p2p
  read (778) nor write (720). Could be the average, or an older test.
- **Correction**: M5 cheatsheet should link to 12_nvlink_p2p §3 and use
  778 (read) or 720 (write), not the unsourced average of 740.

### R5. PCIe full-duplex "1.72× sum" claim (line 17, line 71)
- "H2D + D2H concurrent: 98.8 GB/s = 1.72× sum of either alone".
- Math: 98.8 / (57.5 + 57.1) = 98.8 / 114.6 = **0.86×**, not 1.72×.
- The intended interpretation is "1.72× a SINGLE direction"
  (98.8 / 57.5 = 1.72×). The "sum of either alone" wording is wrong.
- **Correction**: rewrite as "98.8 GB/s aggregate = 1.72× single-
  direction throughput; achieves 86% of the dual-direction sum
  (114.6 GB/s), demonstrating partial-but-not-complete full-duplex
  on Gen 5 effective lanes".

---

## RECONCILIATION (claims that DO check out)

- **PCIe 57.7 GB/s H2D / 57.4 GB/s D2H pinned**: matches three independent
  benchmark binaries (`copy_engines.cu`, `pcie_max_bw.cu`, `pcie_audit2.cu`)
  per §"Files of record". HIGH.
- **`cudaDevAttrAsyncEngineCount = 4`**: API query, deterministic. HIGH.
- **D2D 3279 GB/s @ 2 GB / 3005 GB/s @ 256 MB**: matches
  TRUE_REFERENCE HBM ladder (45% / 41% of 7.30 TB/s peak). HIGH.
- **3.6 µs sync H2D 1 B floor**: matches TRUE_REFERENCE coordination
  ladder line 89 ("`cudaMemcpy` sync (small) | 3.6 µs"). HIGH.
- **Pageable 38 GB/s = 66% of pinned**: defensible; the "1.5 TB/s
  pageable" myth is well-debunked in §"Pageable memory" with the
  page-migration explanation.
- **Power: 200-1100 W via NVML**: matches CLAUDE memory ("1100 W TDP").
  HIGH.
- **`HostNativeAtomicSupported = 0`**: confirms B300 is the pure-PCIe
  variant (not GH200/GB200 NVL with NVLink-C2C). HIGH.
- **`AsyncEngineCount = 4` but no aggregate gain past 1 stream**: well
  established (4 engines share single PCIe link).

---

## UNRESOLVED

1. **Why does PCIe Gen 6 PHY cap at Gen 5 throughput?** — open in §"Open
   questions" item 1. Needs different chassis to isolate (host vs
   switch vs PHY).

2. **Per-GPU vs shared PCIe BW with both GPUs active**: §"Open questions"
   item 2. Single test in `multigpu/` would settle.

3. **GPUDirect RDMA NIC→HBM throughput**: not measured (item 3).

4. **PCIe Gen 6 PAM4 FEC overhead** (item 5): out of scope for CUDA-only
   tooling.

5. **`HostNumaId = 0` interpretation**: 13_pcie reports 1 NUMA node, but
   AMD EPYC 9575F can be configured for NPS1/NPS2/NPS4 in BIOS. The "1
   node" is what BIOS exposes; could be hiding a true NUMA topology.

6. **Effective Gen 6 in any future test**: when (if) a Gen 6 host is
   measured, what's the new effective BW? Currently undefined.

7. **Power coupling between PCIe and NVLink**: under joint H2D + P2P
   load, does either degrade? Untested.

---

## RECOMMENDED CANONICAL NUMBERS (PCIe + System, B300 SXM6 AC)

| Quantity | Value | Notes |
|---|---:|---|
| PCIe link gen / width | **Gen 6 x16** | NVML, lspci confirm |
| PCIe H2D pinned (≥64 MB) | **57.7 GB/s** | 90% of Gen 5 spec, 23% of Gen 6 spec |
| PCIe D2H pinned (≥64 MB) | **57.4 GB/s** | symmetric |
| PCIe full-duplex aggregate | **98.8 GB/s** | 1.72× single-dir |
| PCIe pageable H2D | **38.0 GB/s** | 66% of pinned |
| Async engines | **4** | share single PCIe link |
| D2D same device (2 GB) | **3279 GB/s** | 45% of HBM 7.30 TB/s |
| H2D 1 B sync latency | **3.6 µs** | floor |
| H2D 4 KB async+sync | **6.5 µs** | |
| D2H 4 KB async+sync | **9.0 µs** | reads need ack |
| Persistent kernel + mapped poll | **~4 µs** | best CPU↔GPU RT |
| Power min / max (NVML) | **200 / 1100 W** | not 700, not 1400 |
| Idle baseline | **~180-197 W** | |
| NVLink generation | **NVLink 5** (NV18) | NOT "v7" |
| `HostNativeAtomicSupported` | **0** | pure PCIe variant |
| ECC | **always on** | 1/16 bus reserved |

---

## NAMING DRIFT ACROSS DOCS (PCIe-specific)

| Doc | PCIe spec quoted | Effective | Gap explained? |
|---|---|---|---|
| 13_pcie_system.md | Gen 6 256 GB/s | 57.7 GB/s | Hypotheses, none verified |
| TRUE_REFERENCE.md | "23% of 256 spec" | 0.058 TB/s | Says "CPU-bound" — UNVERIFIED |
| M5_MEMORY_CHEATSHEET.md | not stated | 58 GB/s | implicit |
| CLAUDE.md memory | "PCIe Gen 6 x16, 57.7 GB/s effective" | 57.7 | gap not addressed |

Authoritative answer: PHY runs Gen 6, effective rate caps at ~Gen 5
(57.7 GB/s). **Root cause is unconfirmed**, not "CPU-bound".
