# 12 — NVLink P2P / Multi-GPU — CORRECTIONS

Source-of-truth review of `b300_clean/12_nvlink_p2p.md` against
`B300_TRUE_REFERENCE.md`, `M5_MEMORY_CHEATSHEET.md`, `13_pcie_system.md`,
project memory, and CLAUDE.md.

Originals untouched.

---

## RETRACTIONS

### R1. "NVLink v7" naming in `13_pcie_system.md` line 6 and line 216
- 13_pcie_system.md states "NVLink v7 (NV18, 18 lanes)" — **WRONG**.
- B300 (Blackwell) uses **NVLink 5th-generation (NVLink5)**.
- 12_nvlink_p2p.md §1, §12 retired-claims, and TRUE_REFERENCE line 36
  all correctly call it NVLink-5 / NVLink5.
- CLAUDE.md memory snippet calling it "NVLink v7" is also wrong; Hopper
  H100/H200 = NVLink 4, B100/B200/B300 = NVLink 5. There is no "NVLink 7".
- **Correction**: every "NVLink v7" reference must read **"NVLink 5"**.

### R2. "NV18 = 18 × NVLink5 links per direction" (12_nvlink_p2p §1, line 5)
- nvidia-smi topo nomenclature: `NV18` = **18 NVLink connections**, total
  link count (each link is full-duplex). Not "18 per direction".
- 12_nvlink_p2p §12 retirement table says correctly: "NV18 = 18 NVLink5
  links (i.e. 18 lanes of NVLink generation 5)". The §1 wording
  "per direction" contradicts §12. Internal inconsistency.
- Theoretical math is unaffected because each link is full-duplex
  (50 GB/s per direction), so 18 × 50 = 900 GB/s/dir is correct
  regardless. But the **explanation** is muddled.

### R3. "spec 757" used as denominator in TRUE_REFERENCE lines 36-37
- TRUE_REFERENCE: "NVLink-5 P2P read 0.78 = 1.04× spec 757".
- The 757 GB/s number is **NVLink 4** (Hopper) per-direction
  marketing (18 × 25 GB/s × 1.681 protocol). For NVLink 5,
  per-link is 50 GB/s data → 900 GB/s/dir spec, or 956 GB/s
  raw including protocol overhead (= 18 × 53.125).
- The 12_nvlink_p2p doc itself uses **900 GB/s** as the denominator
  (e.g. line 33: "% of 900 GB/s peak"). 778 / 900 = 86%, not 104%.
- **Correction**: TRUE_REFERENCE should state "0.78 TB/s = 86% of
  900 GB/s NVLink-5 spec" (not 1.04× of NVLink-4 spec).

### R4. Read BW number drift across docs (740 / 778 / 820 / 860)
Same physical quantity reported with 4 different values:

| Source | Value | Method |
|---|---:|---|
| 12_nvlink_p2p §3a | 778 GB/s | event-timed kernel, float4 + ≥75K thr |
| 12_nvlink_p2p §3c | 820 GB/s = 91% of peak | summary line, no method shown |
| 12_nvlink_p2p §11 | 860 GB/s NVLink RX | ncu metric `nvlink__data_received` |
| 12_nvlink_p2p §3d | 792-817 GB/s | SM-count sweep saturation |
| TRUE_REFERENCE line 36 | 780 GB/s (0.78 TB/s) | 4096 blocks v8 (9172429) |
| M5_MEMORY_CHEATSHEET line 28 | 740 GB/s | J2 (#e16901f) |
| Project memory | 820 GB/s read | MGFenceBench |

The 4 values are NOT obviously the same number measured 4 ways. The
**ncu RX (860)** can legitimately exceed the **payload (778)** because RX
counts protocol bytes; that's reconcilable. But §3c's "820 GB/s = 91%"
is not derivable from any other line in the doc. Looks like a hand-typed
estimate that was never reconciled with the §3a measurement table.

**Correction**: pick one canonical number. Recommend
- payload kernel BW: **778 GB/s = 86% of 900 GB/s spec** (matches §3a, §3d, TRUE_REFERENCE within rounding).
- ncu link-level RX: **860 GB/s = 96% of 900 GB/s spec** (includes protocol bytes).
- The "820 GB/s = 91%" claim in §3c is **unsupported** and should be
  retracted or sourced.

### R5. Write BW number drift (710 / 718 / 720 / 765 / 836)

| Source | Value | Method |
|---|---:|---|
| 12_nvlink_p2p §3b | 720 GB/s event-timed | W=128 STG.E.STRONG.SYS |
| 12_nvlink_p2p §3b | 768 GB/s steady state | W=1024 |
| 12_nvlink_p2p §3c | 720 GB/s = 80% of peak | summary |
| 12_nvlink_p2p §3d | 765 GB/s saturated @ 74 SMs | sweep |
| 12_nvlink_p2p §11 | 836 GB/s NVLink TX | ncu |
| TRUE_REFERENCE line 37 | 710 GB/s (0.71 TB/s) | 4096 blocks v8 |
| M5_MEMORY_CHEATSHEET line 28 | 740 GB/s | J2 |
| Project memory | 718 GB/s | MGFenceBench |

§3d shows 765 GB/s at 74 SMs (saturated). §3b headline is 720. §3c
synthesizes 720 = 80%. None of these match TRUE_REFERENCE 710. Spread
is ~125 GB/s = ~17% of the value.

**Correction**: same as R4 — pick a canonical pair (payload + ncu) and
state which is which. Recommend
- payload write: **720 GB/s = 80% of 900 GB/s spec**
- ncu TX: **836 GB/s = 93% of spec** (protocol bytes).

### R6. "Bidirectional 1543 GB/s = 86% × 2" in §2 line 37
- 1543 / 1800 = 85.7% — claim is correct.
- BUT contradicts CLAUDE.md memory which says "757 / 1503". The
  1503 number = 2 × 757 (NVLink-4 spec). For NVLink-5 the bidi spec
  is 1800 GB/s, not 1503.
- **Correction**: CLAUDE.md memory line "NVLink (2× B300): 757 GB/s
  unidirectional / 1503 GB/s bidirectional (NVLink v7)" should read
  "**900 GB/s/dir spec, 1800 GB/s bidi spec; measured 778 unidir / 1543
  bidi at 86%; NVLink 5**".

### R7. NVLink-5 spec claim in §12 retirement: "NVLink BW 757 GB/s uni / 1503 bi: CONFIRMED"
- Self-contradicting. The body uses 900 GB/s as the spec (line 33)
  and 956 GB/s "raw" (line 7). The retirement entry "757 / 1503
  CONFIRMED" is leftover from when this was thought to be NVLink-4.
- **Correction**: rewrite as "NVLink BW 900 GB/s/dir spec uni /
  1800 GB/s bidi: measured 778 / 1543 = 86%."

---

## RECONCILIATION (claims that DO check out)

- **§1 attribute table**: matches `cudaDevP2PAttr*` returns. HIGH.
- **§3d SM-saturation curve**: monotone, sensible, 32 SMs to peak. HIGH.
- **§5 atomic throughput**: LOCAL 49 Gatomic/s + REMOTE 16 Gatomic/s
  matches project memory ("49 LOCAL / 16 REMOTE"). HIGH.
- **§6 fence drain +17.8 K cy**: not contradicted elsewhere. Cross-checked
  against `08_sync_primitives.md`. HIGH.
- **§7 `cudaDeviceEnablePeerAccess` 131 ms cold**: single test (MED), not
  refuted elsewhere. Keep MED.
- **§4a remote atomic 2966 cy at 1920 MHz = 1.55 µs**: matches
  TRUE_REFERENCE coordination ladder line 87 (1662 ns, derived from
  ~3000 cy at boost). Reconciles within 7%.
- **§9 NCCL 10 µs / custom 21 µs all-reduce floor**: not refuted.
- **§10 sharded GEMM 0% slowdown**: surprising but believable (cuBLAS
  L2 tiling). Single test, mark MED.
- **NV18 link count = 18 NVLink-5 links**: matches `nvidia-smi topo -m`
  semantics. HIGH.

---

## UNRESOLVED

1. **Why does §3c summarize READ at 820 GB/s when §3a measures 778?**
   No source visible for the 820 number. Either re-measure or strike it.

2. **TRUE_REFERENCE 757 GB/s spec basis**: where did this denominator
   come from? Likely a holdover from NVLink-4 H100 docs. Need to
   re-derive from NVLink-5 white paper (50 GB/s/link data × 18 links).

3. **NV18 nomenclature**: nvidia-smi shows "NV18". Per NVIDIA docs this
   is "NV-Link with 18 links". Need to confirm with a definitive NVIDIA
   source whether each link is counted as full-duplex (= 18 bidi pipes)
   or half-duplex (= 9 bidi). Current 900 GB/s/dir math assumes the
   former.

4. **836 GB/s ncu TX vs 720 GB/s event payload**: 116 GB/s gap = 14%.
   Likely the ratio of protocol-byte overhead (53.125 raw / 50 data
   = 6%) plus header-per-flit. But the 14% is bigger than 6% — needs
   accounting (maybe ECC bytes counted in TX too).

5. **Cross-GPU latency under contention**: §4a tested with 1 SM warm.
   When all 148 SMs hammer remote, does 1.55 µs hold? Not measured.

6. **NCCL with NVLink-SHARP** — listed as open in §14. Still open.

7. **3+ GPU NVLink topology** — only 2-GPU system tested.

8. **PCIe Gen 6 vs Gen 5 effective** (cross-doc concern, see
   `13_pcie_system_CORRECTED.md`).

9. **`cudaDeviceFlushGPUDirectRDMAWrites`**: §14 says "25 ns ToOwner
   / 886 ns ToAllDevices but not under load" — un-stress-tested.

---

## RECOMMENDED CANONICAL NUMBERS (NVLink-5 P2P, 2× B300 SXM6)

| Quantity | Value | % of 900 GB/s/dir spec |
|---|---:|---:|
| Read payload BW (kernel + DMA, agree) | **778 GB/s** | 86% |
| Read NVLink RX (ncu, includes protocol) | **860 GB/s** | 96% |
| Write payload BW (kernel) | **720 GB/s** | 80% |
| Write NVLink TX (ncu) | **836 GB/s** | 93% |
| Bidi payload aggregate | **1543 GB/s** | 86% (2-direction) |
| SM count to saturate | **32** | — |
| Per-SM unsaturated rate | **~38 GB/s** | — |
| LOCAL atomic Gops/s | **49** | — |
| REMOTE atomic Gops/s | **16** | 33% of LOCAL |
| Cross-GPU atomic latency | **~1.55 µs / ~3000 cy** | 5× LOCAL |
| Cross-GPU fence drain | **+17.8 K cy** | NVLink in flight |
| `cudaDeviceEnablePeerAccess` cold | **131 ms** | one-time |
| `cudaIpcOpenMemHandle` (cross-process) | **56 µs** | — |
| NCCL all-reduce floor | **10 µs** | small msg |
| Custom ring all-reduce floor | **21 µs** | small msg |

---

## NAMING DRIFT ACROSS DOCS

| Doc | Calls it |
|---|---|
| 12_nvlink_p2p.md | "NVLink5" / "NVLink-5" |
| 13_pcie_system.md | **"NVLink v7"** ← WRONG |
| TRUE_REFERENCE.md | "NVLink-5" |
| M5_MEMORY_CHEATSHEET.md | "NVLink (peer)" — no version |
| CLAUDE.md memory | **"NVLink v7"** ← WRONG |

Authoritative answer: **NVLink 5** (5th generation, on Blackwell).
