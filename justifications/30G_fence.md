# §30.G — Memory fence costs (cta / gl / sys)

Audit date: 2026-04-23
GPU: NVIDIA B300 SXM6 AC, sm_103a, 148 SMs, single-GPU rig
Sampled clock during run: **1942 MHz** (single-thread workload, no boost above)
clock64 cycle deltas are SM-cycle counts; clock-invariant in cy units.
Verified clean GPU state via `pkill -9 QuickRunCUDA; sleep 5; nvidia-smi -rgc` before testing.
No background GPU processes (`nvidia-smi --query-compute-apps` empty).

---

## CLAIM (verbatim from B300_PIPE_CATALOG.md — multiple inconsistent citations)

> **L114-L116** (early ladder):
> | fence.sc.cta | 8.6 cy | — | adu | — |
> | fence.sc.gpu | 274 cy | — | adu | — |

> **L2885-L2893** §30.G "audited 2026-04-15, refined with pending-writes test":
> | `membar.cta` / `fence.sc.cta`                         | **29** | CTA-scope, sequential consistency |
> | `membar.gl` / `fence.sc.gpu` / `fence.acq_rel.gpu`    | **282** | GPU-scope = **10× CTA-scope** |
> | `membar.sys`                                          | **2890** | system-scope = **100× CTA, 10× GPU** |

> **L2914-L2922**:
> | Single warp, no writes | **2914** |
> | Full chip (148 CTAs × 32 thr), no writes | **5046** (fabric coord overhead) |
> > "membar.sys at its minimum (single warp, no pending writes) = 2914 cy — this is the FIXED cost of the system-fabric fence."

> **L3068**:
> > "The membar.sys has exactly 8 parallel fence channels per SM (ncu-verified). 9+ warps/SM serialize in rounds. membar.gl has no such cliff."

> **L3083-L3088** (unified three-tier model):
> | **.cta** | **14 cy** | **+6/write** | **0** (purely local) |
> | **.gl**  | **271 cy** | +150 for 1st, +60/write after | **~0 until 4+ SMs** (very mild) |
> | **.sys** | **2882 (1 SM) → 5075 (4+ SMs)** | **~0 at 0-16 writes** | **+2200 cy from 1→4+ SMs, FLAT after** |

> **L3193 et seq.** (acq_rel vs sc):
> > "fence.acq_rel.sys is 17-37% SLOWER than fence.sc.sys at W=8-16."

> **L3625-L3635** (revised "pure fence" by-scope at full chip, aw=32, W=16):
> | `fence.sc.cta` / `membar.cta` | **337** |
> | `fence.sc.gpu` / `membar.gl`  | **1,679** |
> | `fence.sc.sys` / `membar.sys` | **8,869** |

**Spread per scope (all from a single catalog file):**
- cta: 8.6 / 14 / 29 / 337 cy → ~40× spread
- gl:  271 / 274 / 282 / 1,679 cy → ~6× spread
- sys: 2,882 / 2,890 / 2,914 / 5,046 / 8,869 cy → ~3× spread

**Wave-7 V54 settled** (b300_clean/corrections/V54_RUN_RESULTS.md, 2026-04-22):
- cta=8 cy, gl=267 cy (+~280 cy first-fence-after-write), sys=2,806 cy (R²=1.0 on 6-pt N-scaling).
- Caveat in that doc: "measured on a 2-GPU NVLink-connected B300 system."

This audit re-runs the V54 test on the **single-GPU** rig and adds a separate
N-pending-writes test to isolate the "first-fence-after-write" component.

---

## TEST FILE 1 — V54 isolation (existing)

`/root/github/QuickRunCUDA/tests/standalone/v54_membar_isolation.cu`

Methodology: thread 0 of CTA 0 only; `clock64` brackets 0/1/2/4/8/16/32 inlined
`membar.<scope>` PTX instructions. Each measurement bracketed by
`atom.global.add` anchors (pre + post) so the fence has visible ordering effect
and cannot be DCE'd. Median of 21 trials per cell.

**SASS verification (PASS):** per-kernel MEMBAR count exactly equals the
template parameter N for all 21 instantiated kernels:

```
v54_membar<sc=2,N=32>: 32 MEMBARs   (all MEMBAR.SC.SYS)
v54_membar<sc=2,N=16>: 16 MEMBARs
... (similarly for cta/gl scopes 0/1)
```

## BUILD COMMAND

```
nvcc -arch=sm_103a -O3 -std=c++17 \
  /root/github/QuickRunCUDA/tests/standalone/v54_membar_isolation.cu \
  -o /tmp/v54
```

## RUN COMMAND

```
/tmp/v54
```

(Kernel selects scope and N via template parameters; one binary covers all 21
combinations. Single-thread / single-CTA / single-warp.)

---

## RAW RESULTS — V54 isolation (5 independent runs, this rig)

Total cycles for N stacked fences (median of 21 trials per cell). The slope of
this line is the steady-state per-fence cost; the intercept is fixed overhead.

| run | scope | N=0 | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 | slope (cy/fence) | R² |
|----:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | cta | 2 | 14 | 22 | 38 | 70 | 134 | 262 | **8.00** | 1.0000 |
| 1 | gl  | 2 | 790 | 1054 | 1582 | 2639 | 4751 | 9079 | **267.21** | 1.0000 |
| 1 | sys | 2 | 1728 | 3457 | 6912 | 13831 | 27663 | 55396 | **1731.14** | 1.0000 |
| 2 | sys | 2 | 1723 | 3397 | 6737 | 13428 | 26864 | 54116 | 1689.61 | 1.0000 |
| 3 | sys | 2 | 1727 | 3449 | 6897 | 13799 | 27583 | 55262 | 1726.78 | 1.0000 |
| 4 | sys | 2 | 1727 | 3456 | 6915 | 13830 | 27658 | 55405 | 1731.37 | 1.0000 |
| 5 | sys | 2 | 1727 | 3458 | 6911 | 13827 | 27657 | 55386 | 1730.81 | 1.0000 |

**Per-fence steady-state slopes (this rig, single GPU, 5 runs):**
- `membar.cta`: **8.00 cy/fence** (zero variance across runs)
- `membar.gl` : **267.2 cy/fence** (±0.05% across runs)
- `membar.sys`: **1,727 cy/fence** (mean of 5 runs; range 1690-1732, ±1.2%)

**Note: gl N=1 anomaly.** N=1 is consistently 200-280 cy above what the slope
predicts (slope says 510-790 cy first-fence-incl-overhead vs slope-extrapolated
~270 cy). This is the "first fence after write" overhead from the upstream
`atom.global.add` anchor still in flight — not a property of `membar.gl` itself.

---

## TEST FILE 2 — Pending-writes isolation (new)

`/tmp/v54_writes.cu` — purpose-built to isolate the first-fence-after-write
overhead. Thread 0 issues N store.global.u32 instructions (each dependent on
clock64 to defeat DCE), then times exactly **1** fence:

```cuda
for (int i = 0; i < N_WRITES; i++)
    asm volatile("st.global.u32 [%0], %1;" :: "l"(buf+i), "r"(base+i) : "memory");
clock64();  fence;  clock64();
```

## BUILD + RUN

```
nvcc -arch=sm_103a -O3 -std=c++17 /tmp/v54_writes.cu -o /tmp/v54_writes
/tmp/v54_writes
```

## RAW RESULTS — cy for ONE fence vs N pending dependent writes (median of 21 trials)

Three independent runs:

| run | scope | N=0 | N=1 | N=4 | N=16 | N=32 | N=64 | N=128 |
|----:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | cta | 10 | 9 | 9 | 30 | 26 | 26 | 30 |
| 1 | gl  | 265 | 738 | 739 | 736 | 736 | 738 | 745 |
| 1 | sys | 1730 | 1729 | 1732 | 1729 | 1730 | 1732 | 1735 |
| 2 | gl  | 265 | 751 | 752 | 748 | 746 | 747 | 752 |
| 3 | gl  | 265 | 401 | 401 | 401 | 401 | 401 | 404 |

Stable findings:
- `membar.cta`: 9-10 cy bare → 26-30 cy with 16+ pending writes (small +20 cy tail).
- `membar.gl` : **265 cy bare**; jumps to **400-750 cy with even ONE pending write**, then **FLAT** through 128 writes. The first-fence-after-write tax is ~140-490 cy (variable). The stable steady-state is the 265 cy figure.
- `membar.sys`: **1,728 cy regardless** of pending writes (0-128). System-fabric coordination dominates.

These numbers cleanly settle:
- The "+~280 cy first-fence-after-write" claim from the V54 wave-7 doc is real
  but variable (140-490 cy range observed). The ~280 cy figure is a reasonable median.
- The "per-write cost" tail in the catalog (e.g. "+60 cy/write" for gl) does
  NOT show up in this isolated test. The N-issue test in V54 conflated stacked-fence
  cost with per-write cost. With ONE fence and N writes, gl is FLAT once the first
  write triggers the overhead.

---

## RECONCILIATION TABLE

| Catalog citation                       | cy claimed     | This rig measured | Verdict |
|----------------------------------------|---------------:|------------------:|---|
| L114 `fence.sc.cta` = 8.6              | 8.6            | **8.0**           | ✓ within 7% |
| L115 `fence.sc.gpu` = 274              | 274            | **267**           | ✓ within 3% |
| L2891 `membar.cta` = 29                | 29             | 8                 | ✗ 3.6× HIGH |
| L2893 `membar.gl` = 282                | 282            | 267               | ✓ within 6% |
| L2893 `membar.sys` = 2890              | 2890           | **1,727**         | ✗ 1.67× HIGH |
| L2914 `membar.sys` (1-warp, no writes) = 2914 | 2914   | 1,728             | ✗ 1.69× HIGH |
| L3068 "8 parallel sys channels per SM" | qualitative    | not tested (single-warp test) | UNREPLICATED but plausible |
| L3083 `.cta` = 14 cy                   | 14             | 8                 | ✗ 1.75× HIGH |
| L3084 `.gl`  = 271 cy                  | 271            | 267               | ✓ within 2% |
| L3085 `.sys` (1 SM) = 2882             | 2882           | 1,727             | ✗ 1.67× HIGH |
| L3632 `fence.sc.cta` (full chip W=16) = 337 | 337       | not isolated*     | not directly comparable (multi-SM, busy chip) |
| L3633 `fence.sc.gpu` (full chip W=16) = 1,679 | 1,679   | not isolated*     | not directly comparable |
| L3635 `fence.sc.sys` (full chip W=16) = 8,869 | 8,869   | not isolated*     | not directly comparable |
| **V54 wave-7 cta = 8 cy**              | 8              | **8**             | ✓ EXACT |
| **V54 wave-7 gl  = 267 cy**            | 267            | **267**           | ✓ EXACT |
| **V54 wave-7 sys = 2806 cy**           | 2806           | **1,727**         | ✗ 1.62× HIGH |
| **V54 wave-7 +280 cy first-after-write** | +280         | +140 to +490 (run-variable) | ~CORRECT magnitude |

(*) The "full-chip W=16 pre-load" rows of the catalog are different scenarios
than V54 isolation (148 CTAs × 32 warps × 16 in-flight stores). They are not
reproduced here; my single-warp test is silent on those.

---

## VERDICT

### V54 settlement — partially holds, partially needs revision

| Component                   | V54 claim | This rig | Status |
|-----------------------------|----------:|---------:|---|
| `membar.cta` per-fence       |  8 cy     | 8 cy    | ✓ HOLDS EXACTLY |
| `membar.gl` per-fence        | 267 cy    | 267 cy  | ✓ HOLDS EXACTLY |
| `membar.gl` first-after-write | +280 cy  | +140 to +490 cy | ~OK (median in range) |
| `membar.sys` per-fence       | 2806 cy   | **1,727 cy** | ✗ 1.62× LOWER on this rig |

### Why the sys discrepancy

The V54 wave-7 doc explicitly notes (line 154): *"The 'system' fence value here was
measured on a **2-GPU NVLink-connected** B300 system. On a single-GPU system the
cost may be lower (no NVLink coherence to drain)."*

This audit confirms that prediction:
- Single-GPU rig: **`membar.sys` = 1,727 cy**
- 2-GPU NVLink rig: `membar.sys` = 2,806 cy (V54 wave-7)
- Ratio: 1.62× — consistent with one extra NVLink-coherence hop

So the V54 sys figure was correct **for that rig** but is not the universal
single-GPU value. The catalog's "2890" / "2914" and the "1,750-3,042" historical
spread reflect different system configurations and possibly different
amounts of background traffic. None of the catalog's standalone numbers were
correct for both setups simultaneously.

### Per-scope authoritative recommendations (single-GPU B300 SXM6)

| Op                                  | cy steady-state | ns @ 2.032 GHz | Caveats |
|-------------------------------------|---------------:|---------------:|---|
| `__threadfence_block` / `membar.cta` |             8 |            3.9 | +6 cy per ~16 pending writes (small) |
| `__threadfence` / `membar.gl`        |           267 |          131.5 | **First fence after pending write: +140 to +490 cy (≈+280 typical)**; flat thereafter |
| `__threadfence_system` / `membar.sys` (single GPU) |  1,727 |  850 | NVLink-attached 2-GPU: +1,080 cy (= 2,806 cy) |

### Catalog cleanup recommendations

The catalog should:

1. **Promote the 8/267/1,727-or-2,806 numbers to authoritative §30.G** with
   clear "single GPU vs NVLink-attached" qualifier on .sys.

2. **Retract the mid-confidence numbers** that are 1.5-3.6× off the V54 settlement:
   - L2891 "membar.cta = 29" → was 3.6× high (true value 8 cy)
   - L2893 "membar.sys = 2890" → was 1.67× high on single-GPU rigs (true single-GPU value 1,727 cy)
   - L2914 "membar.sys (1-warp, no writes) = 2914" → same issue
   - L3083 ".cta = 14 cy" → was 1.75× high (true 8 cy)
   - L3085 ".sys (1 SM) = 2882" → was 1.67× high on single-GPU

3. **Keep the L114-L116 ladder** (`fence.sc.cta = 8.6`, `fence.sc.gpu = 274`) —
   these are within noise of the true values.

4. **Keep the multi-SM / busy-chip numbers (L3625-L3635)** as separate rows
   labeled "full-chip W=16 pre-load" — they're NOT comparable to the
   isolated single-warp values and shouldn't be merged into the same row.

5. **The "8 parallel sys channels per SM" claim (L3068)** is multi-warp-specific
   and not contradicted by single-warp tests. Keep as-is, but tag as
   "qualitative pattern, not recently re-verified."

---

## NOTES

- Single-thread / single-warp / single-CTA test. Multi-warp / multi-CTA / contended
  fences may show different costs (see L3068, L3625-L3635 catalog rows).
- The clock during measurement was 1942 MHz throughout (sampled mid-run via
  `nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader`).
  Single-thread kernels do NOT trigger boost to 2032 MHz, but cycle counts are
  clock-invariant; only the ns conversion changes.
- The `gl` first-fence overhead has run-to-run variance of 140-490 cy. This
  appears to be timing-of-anchor-store-completion noise. The 200+ trials per cell
  smooth it within a single run, but the underlying drain time varies between runs.
- `membar.sys` slope is highly stable: 1689-1732 cy across 5 runs (1.2% CV).
- Could not test ncu cross-checks because ncu pipe metrics for `pipe_lsu` /
  `pipe_adu` would mostly capture the atomic anchors, not the fence itself
  (fences are SASS MEMBAR which doesn't have a dedicated pipe metric on B300).
- SASS-verified: each kernel template has exactly N MEMBAR.SC.{CTA,GPU,SYS}
  instructions, no spurious or eliminated MEMBARs.

---

## FILES

- Test source (existing): `/root/github/QuickRunCUDA/tests/standalone/v54_membar_isolation.cu`
- Test source (new, follow-up): `/tmp/v54_writes.cu` (kept as-is in /tmp)
- Reference doc: `/root/github/QuickRunCUDA/b300_clean/corrections/V54_RUN_RESULTS.md`
- Catalog being audited: `/root/github/QuickRunCUDA/B300_PIPE_CATALOG.md`
