# HBM3E Denominator — FINAL Settlement (B300 SXM6)

**Date:** 2026-04-22
**Supersedes:** `HBM_DENOMINATOR_RESOLUTION.md` (wave-4) on the "7672 is a ghost"
framing only. All other content of the wave-4 doc (8-stack count, 1/16 ECC,
ncu-payload semantics, migration list) stands.

---

## 1. What changes from wave-4

The wave-4 doc retired **7672 GB/s** as an "arithmetic ghost" produced by mixing
"8 Gbps spec" with "3996 MHz × 2 = 7.992 Gbps" and rounding off 0.1 %. That
framing is **wrong**. Empirical measurement on this device shows:

```
$ nvidia-smi -q | grep -i "memory.*clock"
    Memory                            : 3996 MHz       ← ACTUAL I/O clock
```

7672 is not a ghost — **it is the literal hardware rate** at the I/O clock this
silicon actually runs.

## 2. Recompute from empirical 3996 MHz

```
I/O clock          = 3996 MHz
Per-pin (DDR)      = 3996 × 2     = 7.992 Gbps         (vs spec 8.000 Gbps)
Per-stack (1024-b) = 7992 × 1024/8 = 1022.976 GB/s
8 stacks raw       = 8 × 1022.976 = 8183.8 GB/s         (pre-ECC)
8 stacks post-ECC  = 8183.8 × 15/16 = 7672.3 GB/s       (1/16 SECDED)
```

vs. spec-clean derivation (4000 MHz / 8.000 Gbps):

```
8 stacks raw       = 8192.0 GB/s
8 stacks post-ECC  = 7680.0 GB/s
```

The 8 GB/s gap (7680 − 7672 = 0.10 %) is **real silicon under-spec**, not
arithmetic noise. Common for sustained operation; vendors quote nominal pin
rates and the controller settles at a slightly lower clock domain.

## 3. Both numbers are correct under different framings

| Denominator | Meaning | Use when… |
|---:|---|---|
| **7680 GB/s** | Spec-rated post-ECC at 8.000 Gbps/pin | Comparing across docs, vendors, or vs other GPUs (B200, MI300X). Apples-to-apples vs published peak. |
| **7672 GB/s** | This-device post-ECC at empirical 7.992 Gbps/pin | Asking "how close to what THIS silicon physically can do?" — strictly the right denominator for SoL claims on this GPU. |
| **8192 GB/s** | Spec raw (pre-ECC) | When measurement excludes ECC parity is uncertain (rare; ncu excludes it). |
| **8183.8 GB/s** | This-device raw at 3996 MHz | Symmetric to 7672 on the raw side. |

The 0.10 % gap is below normal run-to-run noise, so for almost every "% of peak"
report it doesn't matter which you cite — but cite ONE consistently.

## 4. 8-stack count: settled by bus width, not capacity

`nvidia-smi` reports **275040 MiB = 268.59 GB** visible memory. Two physical
configs both fit:

- **8 stacks × 12-Hi × 3 GB/die = 288 GB raw** → ×15/16 = 270 GB post-ECC ✓
- **12 stacks × 12-Hi × 2 GB/die = 288 GB raw** → ×15/16 = 270 GB post-ECC ✓

Capacity ALONE cannot distinguish. The settling argument is **bus width**:

- NVIDIA Developer Blog: "16 × 512-bit memory controllers = 8192-bit total"
- HBM3E stack interface: **1024 bits/stack**
- 8192 / 1024 = **8 stacks**

12-stack would require 12288-bit bus, which contradicts NVIDIA's own
publication. **8 stacks confirmed architecturally**, not by capacity arithmetic.
HBM3E does ship in 2/3/4 GB die capacities; B300 uses 3 GB dies.

## 5. Recommended standard

For all `b300_clean/corrections/` and `B300_TRUE_REFERENCE.md` going forward:

- **Default denominator: 7680 GB/s** (spec-comparable, matches B200/MI300X
  conventions, matches NVIDIA's "8 TB/s" marketing rounded for ECC).
- **When SoL precision matters: also cite 7672 GB/s** (this-device actual at
  3996 MHz I/O). Mention both: "98.5 % of 7680 spec / 98.6 % of 7672 actual".
- **Never call 7672 a "ghost".** It is the literal hardware rate.

## 6. Migration delta from wave-4

- `HBM_DENOMINATOR_RESOLUTION.md` §3 row "7672 GB/s … Verdict: Retire — arithmetic
  ghost" → **revise to "Hardware-actual at 3996 MHz I/O; cite alongside 7680
  spec, not in place of it"**.
- `HBM_DENOMINATOR_RESOLUTION.md` §5 second bullet ("Where the 7672 first arose:
  artifact of incorrect intermediate clock estimate") → **revise: "the 1998 MHz
  estimate is correct — `nvidia-smi -q` confirms 3996 MHz I/O on this device"**.
- All other wave-4 conclusions (8-stack, 1/16 ECC, ncu payload-only,
  `01_hbm_bandwidth.md` 12-stack typo) stand unchanged.
