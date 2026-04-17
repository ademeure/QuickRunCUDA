# Investigation 08: True HBM Write Bandwidth on B300 SXM6

**Date:** 2026-04-17  
**GPU:** NVIDIA B300 SXM6 AC, 148 SMs, 2.032 GHz locked  
**Kernel:** `investigations/hbm_write_proper.cu`  
**Methodology:** standalone nvcc binary + QuickRunCUDA, ncu hardware counters (`dram__bytes_write.sum.per_second`), working set = 4 GB (32x L2 capacity), grid-stride loop, SASS-verified

---

## The Contradiction Explained

The B300 catalog contained four different write bandwidth claims:
- "7.09 TB/s" (section 0 summary)
- "3.4 TB/s" (from `st.global.v4.u32` with 296 CTAs x 512 threads)
- "8.5 TB/s" (with 1 GB buffer, persistent grid, v4)
- "7.0 TB/s" (claimed for v8 + 8 CTAs/SM)

**Root cause: every single one was measuring something different, and none used ncu DRAM counters to confirm actual HBM writes.**

1. **3.4 TB/s** — the 296 CTAs x 512 threads kernel with `st.global.v4.u32` was measuring L2 write throughput, not DRAM, because the 1 GB working set fit in L2 (126 MB L2 was cycling through data that got absorbed before reaching HBM).

2. **8.5 TB/s** — same "fire-and-forget" overcount: stores return immediately (kernel completes before writes drain to DRAM), the event timer measures L2 write buffer throughput (~8-14 TB/s) rather than HBM drain rate.

3. **7.09 TB/s (write) and 7.0 TB/s (v8)** — possibly based on the read methodology (ncu `dram__bytes_read` counter was accurate for reads) retroactively applied to writes, or measured via cudaMemset which has different write-combine optimization.

---

## Methodology

- **Working set:** 4 GB (buffer B, 1073741824 dwords), always exceeds L2 capacity (126 MB)
- **Grid:** 296 CTAs (148 SMs × 2 CTAs/SM), 256 threads/block = 75,776 threads total
- **Stride:** grid-stride, each thread's addresses advance by `n_threads × BPT` per iteration
- **ITERS:** chosen so `n_threads × BPT × ITERS ≤ 4 GB` (no overflow), rounded down to UNROLL=8 multiple
- **Anti-DCE:** values depend on `tid ^ iter`, checksum written to C under impossible branch
- **Timing:** CUDA events in standalone binary (not QuickRunCUDA, which shows fire-and-forget artifact)
- **Ground truth:** ncu `dram__bytes_write.sum.per_second` hardware counter, verified `dram__bytes_write.sum ≈ 4.29 GB` per kernel launch

---

## Results (ncu-verified DRAM writes)

| PTX instruction | SASS | DRAM write BW (ncu) | Notes |
|:----------------|:-----|--------------------:|:------|
| `st.global.v4.b32` (default) | `STG.E.128` | **6.20–6.24 TB/s** | Write-allocate; still reaches DRAM at 4 GB WS |
| `st.global.cs.v4.b32` (streaming) | `STG.E.EF.128` | **6.21–6.26 TB/s** | Evict-first; same DRAM BW |
| `st.global.wb.v4.b32` (write-back) | `STG.E.STRONG.SM.128` | **6.07–6.25 TB/s** | Write-back; same DRAM BW |
| `st.global.cs.v8.b32` (v8 stream) | `STG.E.ENL2.256` | **6.14–6.27 TB/s** | "Evict No L2" 256-bit; same DRAM BW |
| `st.global.cs.b32` (scalar stream) | `STG.E.EF` (32-bit) | **6.09–6.15 TB/s** | Slightly lower due to more instructions |
| `st.volatile.global.b32` | `STG.E` (volatile) | **6.07–6.14 TB/s** | Volatile ordering has no extra cost |

**Comparison: HBM read BW (same methodology, ncu-verified)**

| PTX instruction | SASS | DRAM read BW (ncu) |
|:----------------|:-----|-------------------:|
| `ld.global.cg.v4.b32` | `LDG.E.128` | **6.85–6.88 TB/s** |
| `ld.global.cg.v8.b32` | `LDG.E.256` | **6.82–6.89 TB/s** |

---

## Key Findings

### 1. True HBM write peak: **6.1–6.3 TB/s** (ncu-verified)

At 4 GB working set with grid-stride loop and 2 CTAs/SM x 256 threads, all store variants saturate at **6.2 TB/s** sustained DRAM write bandwidth. This is **14% lower than the measured read peak (6.85 TB/s)**.

HBM3E spec for B300 is 8 TB/s bidirectional. The asymmetry (write < read) is expected: HBM3E has higher read than write port utilization due to refresh overhead and write-to-read switching penalties in memory controllers.

### 2. Store width does NOT matter

| Width | SASS | Peak DRAM BW |
|:------|:-----|-------------:|
| 32-bit scalar | `STG.E.EF` | 6.15 TB/s |
| 128-bit (v4) | `STG.E.EF.128` | 6.26 TB/s |
| 256-bit (v8) | `STG.E.ENL2.256` | 6.27 TB/s |

Scalar is marginally lower (more instructions = more issue bandwidth pressure), but v4 and v8 are identical. DRAM bandwidth is the bottleneck, not the LSU instruction width.

### 3. Cache hint does NOT matter at saturation

All cache hints (default, `.cs`, `.wb`, volatile) produce the same DRAM BW within measurement noise. At 4 GB working set size, the DRAM controllers are the bottleneck regardless of cache eviction policy.

### 4. CTAs/SM does NOT matter (1 CTA/SM is already saturated)

| CTAs/SM | n_threads | Peak BW (per-thread-eff) |
|--------:|----------:|--------------------------:|
| 1 | 37,888 | 6.20 TB/s |
| 2 | 75,776 | 6.18 TB/s |
| 4 | 151,552 | 6.15 TB/s |
| 6 | 227,328 | 6.07 TB/s |
| 8 | 303,104 | 6.05–6.11 TB/s (ncu) |

Claim that "8 CTAs/SM reaches 7.0 TB/s" is false; ncu shows 6.1 TB/s at all occupancies.

### 5. Why the catalog contradicts itself: fire-and-forget artifact

When using QuickRunCUDA's warmup-then-time loop:
- Warmup fills the L2 write buffer
- Subsequent timed runs find the L2 already "absorbing" stores
- The CUDA event timer captures kernel completion (when stores enter L2), not DRAM drain
- Result: `12–13 TB/s` "effective" per-thread BW — 2x overcount

**ncu confirms**: the per-thread effective BW from QuickRunCUDA showed 12 TB/s, but `dram__bytes_write.sum` shows only 2–3 GB written (vs 4.29 GB intended), because the remaining data drains to DRAM after the event timer stops.

### 6. The "3.4 TB/s" measurement

The catalog's 3.4 TB/s was measured with `st.global.v4.u32` but with a **1 GB working set** (or possibly with all 148 CTAs × 512 threads overwriting the same small range repeatedly). At 1 GB working set, L2 absorbs all writes — the DRAM drain rate at 3.4 TB/s represents the L2→DRAM write-back throughput, not actual memory bandwidth.

---

## Recommended Write BW Figures for B300

| Scenario | BW | Source |
|:---------|---:|:-------|
| HBM read (sustained, DRAM-bound) | **6.85–6.90 TB/s** | ncu `dram__bytes_read` |
| HBM write (sustained, DRAM-bound) | **6.1–6.3 TB/s** | ncu `dram__bytes_write` |
| Write/read asymmetry | **~0.91×** | write is 9% slower |
| L2 write absorption (small WS) | **~12 TB/s** | per-thread effective (not DRAM) |
| L2 write back to DRAM | **~3–4 TB/s** | at 1 GB WS, all dirty L2 draining |

**Delete the following from the catalog:**
- Rule #12 "DRAM write is half of read BW (3.4 vs 7.3 TB/s)" — WRONG
- "8.5 TB/s DRAM write (v4)" — WRONG (is L2 write buffer, not DRAM)
- "7.0 TB/s with v8 + 8 CTAs/SM" — not confirmed by ncu; actual is 6.1 TB/s

---

## SASS Verification

All store variants produce the expected SASS mnemonics:

| PTX hint | SASS mnemonic | Width |
|:---------|:--------------|------:|
| (default) | `STG.E.128` | 128-bit |
| `.cs` (streaming) | `STG.E.EF.128` | 128-bit |
| `.cs.v8` | `STG.E.ENL2.256` | 256-bit |
| `.wb` | `STG.E.STRONG.SM.128` | 128-bit |
| `volatile` | `STG.E` | 32-bit |

"EF" = Evict-First, "ENL2" = Evict No L2, "STRONG.SM" = write-back with SM-scope coherence.

---

## Run Commands

```bash
# Lock GPU clock
nvidia-smi -lgc 2032

# Build standalone (more reliable than QuickRunCUDA for write BW)
nvcc -arch=sm_103a -O3 -o /tmp/hbm_write3 /tmp/hbm_write3.cu
/tmp/hbm_write3

# ncu verification (run on standalone binary, not QuickRunCUDA)
ncu --metrics dram__bytes_write.sum.per_second,dram__bytes_write.sum,gpu__time_duration.sum \
  --kernel-name "kernel_cs_v4" /tmp/hbm_write3

# QuickRunCUDA (shows fire-and-forget artifact; use ncu to correct)
./QuickRunCUDA investigations/hbm_write_proper.cu \
  -H "#define OP 4" -p -t 256 -b 2 -T 7 -B 1073741824 -0 3536
```
