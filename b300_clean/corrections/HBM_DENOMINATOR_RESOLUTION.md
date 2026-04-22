# HBM3E Denominator Resolution — B300 SXM6

**Date:** 2026-04-22
**Scope:** Settles which HBM3E "theoretical" number to use as the denominator
for all "% of peak" claims in `b300_clean/corrections/`.
**Bottom line:** Standardize on **8 Gbps/pin × 8 stacks × 1024 bit / 8 = 8192 GB/s pre-ECC raw**;
report % against **either 8192 (raw) or 7680 (post-ECC, 1/16 reserved)** with the
denominator stated explicitly. Retire 7672 GB/s (arithmetic ghost; see §3).

---

## 1. The verified HBM3E theoretical for B300 SXM6

NVIDIA's own publications and every credible third-party teardown agree on
**three numbers** for B300 / Blackwell Ultra, all describing the same physical bus:

| Number | Meaning | Source |
|---|---|---|
| **8 TB/s** | Marketing-rounded aggregate bandwidth per GPU | NVIDIA Blackwell Ultra blog; HGX B300 spec; tomshardware; The Register |
| **8192 GB/s** | First-principles raw (pre-ECC) bandwidth | derivation §2 |
| **7680 GB/s** | Post-ECC usable (1/16 reserved for SECDED metadata) | derivation §2; consistent with B200 capacity-side ratio (192 → 180 GB) |

**B300 HBM3E configuration (verified):**
- **8 stacks** of HBM3E, each **12-Hi** (12 dies stacked vertically)
- 16 × 512-bit memory controllers = **8192-bit** total physical bus width
  (equivalent to 8 stacks × 1024-bit per stack)
- **8 Gbps per pin** data rate (same as B200; the B200→B300 jump came from
  taller stacks, not faster pins — bandwidth held flat at 8 TB/s while
  capacity rose 192 → 288 GB)
- **288 GB** total capacity per GPU

**Critical correction to existing docs:** B300 has **8 stacks (12-Hi each)**,
NOT "12 stacks". The "12" everyone keeps quoting refers to **stack height
(dies per stack)**, not stack count. `01_hbm_bandwidth.md` line 136
("12 × HBM3E stacks × 1024-bit = 12,288-bit raw") is **wrong on the stack
count** — and the 7680/12288 = 5/8 ratio it implies for ECC is therefore
nonsense too. The correct accounting is **8 stacks × 1024-bit = 8192-bit raw**,
and the post-ECC overhead matches B200's published 6.25 % (1/16) capacity
reservation.

---

## 2. First-principles derivation

```
Per-stack BW = data_rate × bus_width_per_stack / 8 bits_per_byte
             = 8 Gbps × 1024 bit / 8
             = 1024 GB/s per stack          (= 1.024 TB/s)

Total raw BW = 8 stacks × 1024 GB/s
             = 8192 GB/s = 8.192 TB/s        ← pre-ECC, matches "8 TB/s" rounding

Post-ECC BW (1/16 reserved) = 8192 × 15/16
                            = 7680 GB/s     ← post-ECC usable

Post-ECC BW (1/8 reserved)  = 8192 × 7/8
                            = 7168 GB/s     ← would apply IF stricter parity
                                              (no evidence NVIDIA uses this)
```

**Which ECC overhead?** B200 publishes "192 GB HBM3E raw, 180 GB usable"
= 12/192 = **6.25 % = 1/16**. Same DRAM, same controller family on B300, so
**1/16 is the right ECC ratio**. That gives **7680 GB/s post-ECC**, NOT 7672.

---

## 3. Reconciliation table — each in-use number → what it actually represents

| In-use number | Where it appears | What it actually is | Verdict |
|---|---|---|---|
| **8.0 TB/s** | `CLAUDE.md`, NVIDIA marketing, `V8_HBM_WRITE_SOL.md` | Marketing-rounded raw bandwidth (= 8192 GB/s rounded down to 1 sig fig) | **Correct as colloquial; use 8192 GB/s (or 8.19 TB/s) when precision matters.** |
| **8192 GB/s / 8.19 TB/s** | derivation; `01_hbm_bandwidth.md` retired-claims row mentions 8183.8 | True pre-ECC raw bandwidth | **Correct denominator if reporting against raw bus capability.** |
| **7680 GB/s** | implied by `01_hbm_bandwidth.md`'s "ECC reserves 1/16" sentence | Post-ECC usable bandwidth | **Correct denominator if reporting against bandwidth available to user data after SECDED.** |
| **7672 GB/s** | `01_hbm_bandwidth.md` line 138, `B300_TRUE_REFERENCE.md` line 25, `corrections/HBM_INCONSISTENCY_LOG.md` row 1 | Arithmetic ghost: `7680 × 3996 MHz × 2 / 8 / 1e9` recomputed with a 3996 MHz / "8 Gbps" mismatch. The derivation says "datasheet 8 Gbps/pin" but plugs in 3996 MHz × 2 = **7.992 Gbps** (the physical I/O clock × 2 for DDR), losing 0.1 % to rounding. | **Retire — same intent as 7680 but off by 8 GB/s. 7680 is cleaner and matches the 1/16 ECC ratio exactly.** |
| **7.31 TB/s** | `V32_V40_FINDINGS.md`, `V41_V48_FINDINGS.md` "98.5 % of HBM peak" | **Empirical pure-direction ceiling on this specific GPU** — the best wall-clock + ncu-verified read peak (a04d9c8 / NINJA recipes). It is a measurement, not a theoretical. | **NEVER use as denominator for "% of theoretical".** Reserve for "% of SoL recipe" comparisons; always disclose denominator. |
| **7.20 TB/s** | `V8_HBM_WRITE_SOL.md` ("HBM3E spec"); `V46` "98.5 %" | Same family of empirical numbers; lower-quality recipes. Calling this the "spec" is wrong. | **Retire as a "spec" denominator.** |
| **8183.8 GB/s** | retired in 01 line 142 | Pre-ECC raw computed with the same 3996×2 mismatch (8192 × 7992/8000) | **Already correctly retired; equivalent to 8192 within rounding noise.** |

---

## 4. Recommendation: standardize all "% of peak" in corrections/ on this denominator

**Rule:** Every HBM3E % claim must state which of the three theoretical anchors
it is normalized against. Use the canonical names below and the canonical numbers.

| Anchor name | Number | When to use |
|---|---:|---|
| **HBM raw (pre-ECC)** | **8192 GB/s** (= 8.19 TB/s) | When comparing against the physical bus ceiling — i.e. "what the hardware could do if every byte on the wire were user payload". This is the architecturally honest "100 %". |
| **HBM post-ECC** | **7680 GB/s** (= 7.68 TB/s) | When comparing against bandwidth available to user data after the controller reserves 1/16 for SECDED. Use this if a measurement reflects only payload bytes (e.g. ncu's `dram__bytes_read.sum` does NOT include ECC parity, so this is the right denominator for ncu-derived numbers). |
| **HBM SoL recipe** | **7.31 TB/s** (B300 measured pure-R or pure-W ceiling) | When comparing one recipe to another's empirical ceiling on this specific silicon. NEVER call this "theoretical" or "spec". Always cite the commit (a04d9c8 NINJA). |

**Concrete migration from current state:**
- `B300_TRUE_REFERENCE.md` line 25 "7.30 TB/s = 95 % of 7672" → **change to "7.30 TB/s = 95.0 % post-ECC (7680) / 89.1 % raw (8192)"**.
- `01_hbm_bandwidth.md` lines 136–142 — fix "12 × stacks" to "8 × stacks (12-Hi each)", change 7672 → 7680, drop the spurious ×3996 MHz arithmetic.
- `V41_V48_FINDINGS.md` "7.20 TB/s = 98.5 %" — denominator is the empirical 7.31 ceiling. **Restate as "98.5 % of empirical pure-R recipe (a04d9c8); 93.75 % post-ECC; 87.9 % raw"** so the headline is no longer misleading.
- `CLAUDE.md` "matches 8 TB/s spec" — keep the colloquial "~8 TB/s" but add: "= 8192 GB/s raw / 7680 GB/s post-ECC".

**ncu interpretation note:** `dram__bytes_read.sum` from Nsight Compute counts
**user payload bytes only** (the ECC parity bytes are inside the controller and
not exposed). So an ncu-measured 7.30 TB/s should be normalized against the
**7680 post-ECC** denominator, giving 95.0 %, not against 8192 (which would
double-discount and give 89.1 %). This is why the existing 95 % numbers feel
right — they are correctly normalized; only the **printed denominator (7672)
is one digit off** from the architecturally correct **7680**.

---

## 5. Open ambiguity (documented, not resolved)

- **Per-pin spec for B300 specifically:** every NVIDIA-side source quotes B300
  as "8 TB/s" without breaking out per-pin Gbps. The 8 Gbps/pin figure comes
  from the B200 datasheet; B300 holds bandwidth flat while changing only stack
  height, so 8 Gbps/pin is the only consistent value. JEDEC HBM3E officially
  ranges 8.0 – 9.6 Gbps/pin; NVIDIA chose the conservative 8.0 end. **This is
  inferred, not directly cited from a B300 datasheet line item.**
- **Where the 7672 first arose:** it is mathematically `7680 × (7992/8000)` —
  i.e. someone derived it as `7680 bit × (1998 MHz × 2) × 2 / 8 / 1e9` using
  a presumed 1998 MHz I/O clock that gives 3.996 GT/s instead of 4.0. No
  NVIDIA source quotes 1998 MHz for B300; 8 Gbps/pin = 4.0 GT/s × 2 (DDR) is
  the clean spec. **The 7672 number is an artifact of an incorrect intermediate
  clock estimate** and should be retired in favor of the round 7680.

---

## 6. Citations

- [Inside NVIDIA Blackwell Ultra (NVIDIA Developer Blog, updated 2025-09-24)](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) — "Eight 12-Hi stacks", "16 × 512-bit controllers (8,192-bit total width)", "288 GB HBM3e", "8 TB/s per GPU"
- [The Register: Nvidia unveils 288 GB Blackwell Ultra GPUs (2025-03-18)](https://www.theregister.com/2025/03/18/nvidia_blackwell_ultra/) — "Eight stacks", "12-high modules", "288 GB", "8 TB/s"
- [Tom's Hardware: Blackwell Ultra B300 announcement](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-blackwell-ultra-b300-1-5x-faster-than-b200-with-288gb-hbm3e-and-15-pflops-dense-fp4) — Confirms 288 GB / 8 TB/s, capacity bump via 12-high stacks
- [Glenn K. Lockwood — B300 garden page](https://www.glennklockwood.com/garden/processors/B300) — "288 GB HBM3e (8 stacks, 12-high)", "8 TB/s (max)"
- [Civo: B200 vs H100](https://www.civo.com/blog/comparing-nvidia-b200-and-h100) — Confirms B200 192 GB raw → 180 GB usable (= 6.25 % = 1/16 ECC)
- [Wikipedia: High Bandwidth Memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory) — HBM3E per-pin range 8.0 (SK Hynix) – 9.6 (Micron) Gbps
- [VideoCardz / Spheron / Hyperstack / Acecloud / Introl / server-parts.eu Blackwell Ultra writeups](https://videocardz.com/newz/nvidia-blackwell-ultra-gb30-features-20480-cuda-cores-288gb-hb3e-memory-and-pcie-gen6) — All independently confirm 8 TB/s / 288 GB / 12-Hi.
- Internal: `b300_clean/01_hbm_bandwidth.md` lines 134-142 (existing derivation, with errors corrected here)
- Internal: `b300_clean/B300_TRUE_REFERENCE.md` line 25
- Internal: `b300_clean/corrections/HBM_INCONSISTENCY_LOG.md` (this resolution closes its row 1, "Three denominators for % of peak")
- Internal: `b300_clean/V41_V48_FINDINGS.md` lines 9, 18, 85 (the "98.5 %" headlines that need restating)
