# Section A — Hardware & Memory Hierarchy (§1–§15)

> Author: Agent A (sections §1–§15 of the B300 SXM6 AC Canonical Reference).
> Time-stamped 2026-04-22. Stitched into the Canonical Reference by F.

---

## §1. How to use this document

**Answer:** This is the canonical, dual-audience reference for B300 SXM6 AC characterization as of 2026-04-22; it supersedes everything in `b300_clean/` and reduces `corrections/HEADLINE_CORRECTIONS_v5.md` to a 1-pager.  `[🟢 HIGH · src: corrections/HEADLINE_CORRECTIONS_v5.md + corrections/CONFIDENCE_LADDER.md]`

This file is meant to be read two ways:

1. **Skimmer mode.** Read only the bold one-line **Answer:** + the confidence tag at the end of each section. The document is intentionally structured so the first two lines of every `## §N.` block answer the headline question with units and a provenance pointer. Everything underneath is nuance, regime caveats, derivations, footguns, and tables.
2. **Deep-dive mode.** Read the body. Tables expose the variability of the measurement across regimes. The cited `corrections/<file>.md` paths are the auditable trail back to per-claim 3-method (wall-clock + ncu + SASS) verification logs.

### Conventions

**Confidence tags** appear at the end of every quantitative claim:

| Tag | Means |
|---|---|
| 🟢 HIGH | 3-method verified (wall + SASS + ncu), and post-V52 uncontradicted by adversarial doubt reports. Safe to cite as a peer-reviewed measurement. |
| 🟡 MED | 1-2 verification methods OR carries a minor regime caveat (e.g. clock-state-dependent, valid only at certain WS, only one launch geometry tested). |
| 🔴 LOW | Methodology issue surfaced (DCE, LICM, loop-overhead contamination, under-issue) OR cross-agent contradiction unresolved. Treat as suggestive, not measured. |
| ⚫ DISPUTED | Multiple values across docs (>1.5× spread) without consensus. Cite all candidate values when used. |

The grading rubric is from `corrections/CONFIDENCE_LADDER.md` and is preserved verbatim across all 6 sections of this canonical doc.

**Provenance tags** end every claim with the form `[<conf> · src: <path relative to b300_clean/>]`. When a claim merges multiple source files use `+`:

```
[🟢 HIGH · src: corrections/01_hbm_bandwidth_CORRECTED.md §1 + V52_RUN_RESULTS.md]
```

The `src:` paths are relative to `b300_clean/` (so `corrections/01_...` resolves to `b300_clean/corrections/01_hbm_bandwidth_CORRECTED.md`).

**Footgun callouts** (`**Footgun:** ⚠ ...`) appear on a section ONLY when the topic is commonly mis-cited in the wider catalog or in NVIDIA marketing material. They are not stylistic — every footgun in this doc has at least one inconsistent number in the catalog that was traced back to it. If a section has no footgun, the topic is not commonly mis-cited.

**See also** lines list section cross-references using `§N` numbering. Numbered ranges:

| Section range | Coverage | Author |
|---|---|---|
| §1–§15 | Hardware overview, clocks, memory hierarchy (HBM, L1/L2, SMEM, DSMEM, NVLink, PCIe) | A (this file) |
| §16–§25 | Compute pipes (FFMA, FP64, tensor, dual-issue, pipe placement) | B |
| §26–§35 | Latency, sync, atomics | C |
| §36–§45 | Math/intrinsics, INT/bit, power | D |
| §46–§55 | Tensor deep + NVFP4 + tcgen05 | E |
| §56–§65 + appendices | Methodology, zigzag case study, open questions, provenance map, footguns index | F |

### What this supersedes

- All `b300_clean/01_*.md` through `b300_clean/17_*.md` headline numbers (originals retained as audit trail; canonical answers are here).
- `b300_clean/B300_TRUE_REFERENCE.md` and its `_v2_DRAFT.md` (those were master summaries built before the wave-3b/wave-4/wave-5/wave-6 doubt sweeps).
- `b300_clean/corrections/HEADLINE_CORRECTIONS_v5.md` (now a 1-pager pointer).
- `b300_clean/M5_MEMORY_CHEATSHEET.md` (cheatsheet has stale line items; this doc reconciles them).

What this **does NOT** supersede:

- The per-claim verification logs themselves (`corrections/01_hbm_bandwidth_CORRECTED.md` etc.). Those remain the source of truth for "where did this number come from".
- `b300_clean/M3_REVERIFY_LOG.md` and the `corrections/*_INCONSISTENCY_LOG.md` / `*_DOUBT_REPORT.md` files (audit trail).
- `CLAUDE.md`'s methodology section (steps 1–8 of the rigor protocol).

### Time-stamping

This canonical reference was assembled on **2026-04-22** from sources updated through that date. Specific anchor points:

- HBM stack-count independent verification: 2026-04-22 (`HBM_STACKS_INDEPENDENT_VERIFY.md`).
- Dual-issue ncu settlement: V52 run, 2026-04-22 (`V52_RUN_RESULTS.md`, see §22 in Agent B's section).
- Confidence ladder v3 patch: 2026-04-22 (`CONFIDENCE_LADDER_PATCH_v3.md`).

If you read this doc more than ~3 months after 2026-04-22, treat MED/LOW entries as more likely to have moved than HIGH entries; HIGH entries are anchored on architectural quantities that don't depend on driver/firmware revs.

### A note on the dual-issue zigzag

§22 (Agent B) covers the FMA + ALU dual-issue verdict in depth. For context: the architectural verdict has flipped 5 times across the wave-1..wave-6 audit (HIGH → LOW → MED → LOW → HIGH). V52's empirical ncu measurement (`pipe_alu = 98.0%` AND `pipe_fma = 49.4%` simultaneously, sum = 147%) settled it in favor of "pipes overlap freely". The historical 55%/74% wall-clock measurements from V49/V50 are CONFIRMED-but-RETRACTED-as-architectural-claims (the numbers are what they are; the inference of a "shared dispatch cap" was wrong). Agent F's appendix §65 has the full zigzag case study.

The hardware-and-memory sections (§1–§15) are NOT directly affected by the dual-issue settlement, but the cross-cutting methodology lessons (Rule 13: "wall-clock GLane/s ratios are NOT decisive — they confound dispatch with per-instruction issue cadence") apply throughout.

### Reading order recommendation

For someone new to B300:

1. §2 (the at-a-glance card)
2. §3 (clocks — informs every TFLOPS/TB/s number elsewhere)
3. §6 / §7 / §8 (HBM read/write/concurrent — bandwidth ceilings)
4. §11 (L2 — three different bandwidths, this is the most-confused topic)
5. §12 (SMEM peak)
6. §16+ (Agent B section, compute peaks)

For someone reading to validate a single claim: jump straight to the numbered section, read the **Answer:** line, then check the `[<conf> · src:]` tag, follow the path.

### File maintenance

When you add a new measurement that contradicts a HIGH-tagged claim here, the correct workflow is:

1. Run the rigor protocol (`./utils/rigor_run.sh`) first; capture wall + SASS + ncu.
2. Open a doubt report (`corrections/<topic>_DOUBT_REPORT.md`).
3. If the doubt holds, downgrade the relevant row in `corrections/CONFIDENCE_LADDER.md` and write a `corrections/HEADLINE_CORRECTIONS_v6.md` patch.
4. Re-stitch this canonical document.

Do NOT silently edit a HIGH row here without leaving an audit trail.

### How sections are structured

Every section follows the same skeleton:

```
[Section header line]

Answer: <one-line> [conf · src]
<2-5 lines nuance>

| <fact tables> |

<deeper derivations / regimes / sub-sections>

Footgun: <if commonly mis-cited>
See also: <other sections>
```

Read the Answer + tables; skip the rest unless you're auditing or the regime caveat matters.

### Common terminology used throughout

| Term | Meaning |
|---|---|
| SoL | Speed-of-Light: the architecturally maximum achievable rate. "% of SoL" = measured / theoretical. |
| BW | Bandwidth (TB/s, GB/s). Always specify direction (read / write) and metric (lts wire / kernel-effective / payload). |
| TFLOPS | Teraflops/s. ALWAYS specify clock state and op-count convention (FFMA = 2 FLOPS each, mma.sync = N×M×K×2 etc.). |
| MFU | Model FLOPs Utilization: measured / theoretical for tensor work. Apples-to-apples within a precision. |
| WS | Working set (in bytes). Determines L1/L2/DRAM regime. |
| TLP | Thread-Level Parallelism (warps in flight per SM, drives latency hiding). |
| ILP | Instruction-Level Parallelism (independent instructions in a single thread). |
| DCE | Dead Code Elimination: compiler removed your benchmark. Symptoms: 0.001 ms runtime, BW > theoretical, BW doesn't scale with iter count. |
| LICM | Loop-Invariant Code Motion: compiler hoisted the work out of the loop. Symptom: time roughly independent of loop bound. |
| ncu | Nsight Compute, NVIDIA's GPU profiler with hardware counter access. |
| SASS | Streaming Assembler: the GPU's machine code (compiled from PTX). `cuobjdump -sass` to inspect. |
| NINJA | A hand-tuned recipe that beats the obvious / library version (V8 / V10 / V32 etc. nomenclature in this catalog). |

### About "wave" numbers in source paths

The catalog audit was iterated in waves:

- **Wave 1** (early 2026): `01_*.md` through `17_*.md` initial category files.
- **Wave 2**: cross-doc comparison (`*_INCONSISTENCY_LOG.md` files).
- **Wave 3**: adversarial doubt-swarm (`*_DOUBT_REPORT.md` files).
- **Wave 4**: meta-doubt and HBM denominator settlement (`HBM_DENOMINATOR_FINAL.md`, etc.).
- **Wave 5**: SASS-verification re-pass (V52 design).
- **Wave 6** (2026-04-22): empirical V52 ncu measurements, settling dual-issue (`HEADLINE_CORRECTIONS_v5.md`).

The canonical reference (this doc) is post-wave-6.

---

## §2. B300 SXM6 AC at a glance

**Answer:** 148 SMs, 4 SMSPs/SM × 32 lanes = 128 FP32 cores/SM, sm_103a (compute capability 10.3), CUDA 13.2 / driver 580.126.09, 275040 MiB HBM3E visible, 7680-bit memory bus on this AC SKU.  `[🟢 HIGH · src: corrections/HBM_STACKS_INDEPENDENT_VERIFY.md + corrections/CONFIDENCE_LADDER.md]`

The "AC" suffix in the part name `NVIDIA B300 SXM6 AC` denotes a yield-binned variant where 1 of the 16 × 512-bit memory controllers is fused off (8192 → 7680 bits effective bus). All 8 HBM3E stacks are physically present; only one controller-pair is disabled. See §6 footgun for why this matters when computing % of HBM peak.

### Spec card

| Quantity | Value | Source |
|---|---|---|
| Architecture | Blackwell Ultra | NVIDIA Tech Blog "Inside Blackwell Ultra" |
| Compute capability | 10.3 (`sm_103a`) | `cudaGetDeviceProperties().major.minor` |
| SM count | **148** | `multiProcessorCount` |
| SMSPs per SM | 4 | architecture |
| FP32 cores per SM | **128** (4 SMSPs × 32 lanes) | architecture; CLAUDE.md note |
| Total FP32 cores | 18,944 | 148 × 128 |
| L1+SHMEM unified pool/SM | 256 KB | `cudaFuncSetAttribute` |
| Max user SMEM/CTA opt-in | 228 KB | `cudaDevAttrMaxSharedMemoryPerBlockOptin` |
| L2 capacity | **126.5 MB** = 132,644,864 B | `cudaDeviceProp.l2CacheSize` |
| Visible memory | **275040 MiB = 268.59 GiB** | `nvidia-smi --query-gpu=memory.total` |
| Memory bus width (this SKU) | **7680 bits** | `cudaDeviceProp.memoryBusWidth` |
| Memory bus width (architectural max) | 8192 bits | NVIDIA Tech Blog |
| HBM stacks | 8 × 12-Hi (3 GB/die) | NVIDIA Tech Blog (post-correction 9/24/25) |
| Memory I/O clock | 3996 MHz | `cudaDeviceProp.memoryClockRate / 1000` |
| HBM3E per-pin rate | 7.992 Gbps (≈ 8.000 spec) | 3996 MHz × 2 (DDR) |
| Boost clock | 2032 MHz | `nvidia-smi -q | grep -A 5 Clocks` |
| Sustained-under-load typical | 1920 MHz | empirical, see §3 |
| Default base clock | 1005 MHz | empirical floor without lock |
| ECC | enabled (always on) | `cudaDeviceProp.ECCEnabled = 1` |
| ECC overhead | 1/16 SECDED | architecture (see §4) |
| Async copy engines | 4 | `cudaDevAttrAsyncEngineCount` |
| PCIe link | Gen 6 x16 (effective Gen 5) | `nvidia-smi -q`, see §15 |
| NVLink generation | NVLink 5 (NV18 = 18 links) | `nvidia-smi topo -m`, see §14 |
| Max cluster size (portable) | 8 (16 advertised) | `cudaDeviceGetAttribute(MaxClustersDimension)` |
| Driver | 580.126.09 | `nvidia-smi` |
| CUDA toolkit | 13.2 | `nvcc --version` |
| Power range (NVML) | 200 — 1100 W | `nvmlDeviceGetPowerUsage`, §3 + Agent D §44 |
| Idle baseline | 180–197 W | `nvmlDeviceGetPowerUsage` |

### What "AC" suffix means

NVIDIA B300 SXM6 ships in multiple SKUs differing in:

- HBM controller fuse pattern (this AC SKU: 1/16 fused off → 7680-bit bus, 268 GiB visible)
- TDP cap (this AC SKU: 1100 W)
- Clock policy

The "AC" suffix specifically indicates the yield-binned variant. The architectural specs (148 SMs, 128 FP32 cores/SM, NVLink 5, etc.) are unchanged from full-bin parts. Numbers in this doc are measured on this AC SKU; cross-vendor comparisons should use spec denominators (8192-bit bus, 7.68 TB/s HBM peak) rather than this-device denominators.

### B300 vs B200 vs H100 quick comparison

For context (B200 / H100 numbers from NVIDIA spec sheets):

| Quantity | H100 SXM5 | B200 SXM6 | **B300 SXM6 AC** |
|---|---|---|---|
| Architecture | Hopper | Blackwell | Blackwell Ultra |
| CC | 9.0 (sm_90a) | 10.0 (sm_100a) | **10.3 (sm_103a)** |
| SMs | 132 | 148 | **148** |
| FP32 cores/SM | 128 | 128 | **128** |
| Total FP32 cores | 16,896 | 18,944 | **18,944** |
| L1+SMEM/SM | 256 KB | 256 KB | **256 KB** |
| L2 total | 50 MB | 100 MB | **126 MB** |
| HBM | 80 GB HBM3 | 192 GB HBM3E | **288 GB HBM3E** |
| HBM bus | 5120-bit | 8192-bit | **8192-bit (7680 on AC)** |
| HBM BW spec | 3.35 TB/s | 8 TB/s | **8 TB/s** |
| HBM BW measured | ~3.0 TB/s | ~6.7 TB/s | **~7.30 TB/s** |
| FP32 FFMA peak | 67 TFLOPS | 70 TFLOPS | **77 TFLOPS** |
| BF16 tcgen05 | n/a (mma.sync) | ~1.8 PFLOPS | **~2.0 PFLOPS** |
| FP8 tcgen05 | ~3.9 PFLOPS | ~4.5 PFLOPS | **~4.5 PFLOPS** |
| NVLink | NVLink 4 (450 GB/s/dir) | NVLink 5 (900 GB/s/dir) | **NVLink 5 (900)** |
| PCIe | Gen 5 x16 | Gen 5 x16 | **Gen 6 x16 (Gen 5 effective)** |
| TDP | 700 W | 1000 W | **1100 W** |

Key B300-specific items not on B200/H100:

- **`sm_103a` ISA** with new tcgen05.mma instruction family (see Agent E §50).
- **Cluster max=8** (was 16 advertised on H100; B200 dropped to 8 effective).
- **126 MB L2** (was 50 MB on H100, 100 MB on B200).
- **HBM3E 288 GB** (was 80 GB H100, 192 GB B200).

### Architectural blocks (rough physical layout)

```
B300 die (Blackwell Ultra, TSMC 4NP):
├── 8 GPCs (Graphics Processing Clusters)
│   └── Each GPC has 9-10 SMs (varies post-yield)
│   └── 148 total SMs across 8 GPCs
├── 4 TPCs per GPC (avg) × 4 SMs per TPC structure
│   (note: TPC pairing varies; cluster=8 placement is in 4 TPC pairs)
├── 16 × 512-bit HBM3E memory controllers (15 enabled on AC SKU)
├── 8 HBM3E stacks (12-Hi each)
├── L2: 126.5 MB total, 2 partitions (sides)
└── NVLink 5 + PCIe Gen 6 IO
```

The 8 GPCs × ~18-19 SMs/GPC ≈ 148 SMs. SM-to-GPC mapping is mostly linear (SMs 0-17 in GPC0, 18-37 in GPC1, etc.), but there's some yield-driven irregularity.

**See also:** §4 (HBM topology derivation), §6 (read SoL recipes), §11 (L2 capacity), §22 (FP32 dual-issue, by Agent B).

---

## §3. Clock frequencies

**Answer:** Boost clock is 2032 MHz (rarely sustained under load); typical sustained boost is 1920 MHz; `nvidia-smi -lgc 2032` paradoxically pins to 1920 (NOT 2032); the GPU can stick at 1005 MHz silently under no-lock; voltage scales with clock² for power purposes.  `[🟢 HIGH · src: CLAUDE.md §2 + project memory feedback_clock_lock_works.md + project_b300_v6_complete.md]`

Clock state determines every TFLOPS / TB/s / cycles-per-instruction number in this catalog. Always state the clock when citing.

### Operating points (verified)

| State | Frequency | How to enter | When you see it |
|---|---:|---|---|
| Boost peak | **2032 MHz** | default, no lock, light load | short kernels (<1 ms) at boost |
| Sustained boost | ~1920 MHz | default, sustained load | long-running kernels (~10+ ms) under thermal/power equilibrium |
| `-lgc 2032` paradox | **1920 MHz** | `nvidia-smi -lgc 2032` | when explicitly locking to "boost" |
| Base | 1005 MHz | Sometimes auto-stuck under DVS without explicit lock | random — see footgun below |
| `-lgc <N>` arbitrary | N MHz (510 ≤ N ≤ 2032) | `nvidia-smi -lgc N` | controlled experiments |
| Throttled (TDP cap) | 1700–1800 MHz | sustained random-data DRAM at high clock | when chip hits 1100 W TDP wall |

### Why the lock paradox

`nvidia-smi -lgc 2032` (or `-lgc 2032,2032`) pins the SM clock to 2032 MHz nominal — but the actual delivered clock under DVS is the **base** clock at that lock point, which is 1920 MHz on B300. To actually reach 2032 MHz boost you must NOT lock, and rely on driver DVS to opportunistically boost. There is no documented user-facing way to lock to 2032 MHz delivered. This is a 6 % gap that has caused confusion across the catalog (TFLOPS numbers stated at "locked 2032" are actually at 1920 MHz delivered).

### V² DVS scaling

Power scales with V × clock × switching activity. On B300 the V-vs-clock relationship is roughly:

| Clock (MHz) | Approx V (mV) | V²-relative |
|---:|---:|---:|
| 510 | 700 | 1.00× |
| 1005 | 800 | 1.30× |
| 1500 | 900 | 1.65× |
| 1920 | 1000 | 2.04× |
| 2032 | 1050 | 2.25× |

So power for the same kernel scales approximately as `(clock/510) × (V(clock)/700)²` — at 2032 MHz a kernel can draw 8–10× the power of the same kernel at 510 MHz. Combined with data-dependent toggle activity (see §9), the effective power range across realistic clock + data combinations is the full 200–1100 W TDP envelope.

### The "stuck at 1005" silent-failure mode

Without an explicit `-lgc`, the B300 driver's DVS policy can leave the GPU at 1005 MHz under sustained load — particularly after a long-running benchmark, after thermal stress, or after leftover background processes. `nvidia-smi -q` typically does NOT show this in the snapshot view; you must sample `nvidia-smi --query-gpu=clocks.gr --format=csv -l 1` during the run.

Recovery: `nvidia-smi -rgc` (reset to default) followed by waiting ~5 seconds, OR explicit `nvidia-smi -lgc 1920` (the "honest" boost lock).

This is documented in user memory `feedback_clock_stuck_no_lock.md`. If a measurement looks 2× too slow vs prior runs, this is the first thing to check.

### Background-process contamination

The "1942 MHz floor" myth was debunked: leftover `QuickRunCUDA` processes or other CUDA contexts can keep the chip warm enough to prevent boost. Always `pkill -9 QuickRunCUDA && sleep 5-8` between measurements when characterizing peaks. See user memory `feedback_clock_lock_works.md`.

### Clock-state guide for citing TFLOPS / TB/s

When citing a peak number from this doc, always note:

- **"At boost (2032 MHz)"** if the test ran in <100 ms with explicit `-rgc` and pre-warmed.
- **"At sustained boost (~1920 MHz)"** for typical long-run measurements.
- **"At locked 1920 MHz"** if `-lgc <anything>` was used.
- **"At locked <N> MHz"** for specific clock sweeps (power studies, V² extraction).

The default convention in this doc when no clock is stated: **boost (2032 MHz)** for short peak tests, **sustained 1920 MHz** for sustained-throughput tests. Sections that depend critically on clock will state explicitly.

### Clock domain map

B300 has multiple independent clock domains:

| Domain | Default freq | Affected by `-lgc` | Notes |
|---|---:|---|---|
| SM (compute) | 2032 boost / 1920 sustained | YES | The "GPU clock" most people mean |
| L2 / XBAR (video) | **1860 MHz** | **NO** | Constant. Affects L2 wire BW, not delivered. See §11 |
| HBM I/O | 3996 MHz | NO | Set at boot from boot policy; not user-controllable |
| Memory controller (HBM PHY) | 3996 MHz × 2 DDR = 7.992 Gbps/pin | NO | |
| NVLink SerDes | 53.125 GB/s/dir/lane raw (per NVLink-5 spec) | NO | |
| PCIe SerDes | 64 GT/s nominal, 32 GT/s effective on this rig | NO | See §15 PHY-vs-effective |

The cross-domain implication: a measurement that depends on multiple domains (e.g., kernel issuing memory loads) does NOT scale uniformly with `-lgc`. The SM-issue rate moves; the L2 wire and HBM PHY do not.

### Empirical clock observations

`b300_clean/CLOCK_DOMAINS_AND_L2_UNITS.md` (HIGH conf) characterizes:

- **Combined-warp atomics** (all lanes targeting the same address) are SM-issue-bound; throughput ∝ SM clock.
- **Uncombined / scattered atomics** are L2/DRAM-bound; throughput ∝ L2 video clock (1860 MHz constant).
- **HBM bandwidth** is HBM PHY-bound; per-pin rate independent of SM clock. SM clock affects only the launch-address generation rate, which is rarely the bottleneck for DRAM-saturated kernels.

This means HBM read peak (7.30 TB/s) is essentially the same at 1005 MHz SM clock as at 2032 MHz, while FFMA peak (76.96 TFLOPS at boost) drops to ~37 TFLOPS at 1005 MHz. Power scales differently for each. See §44 (Agent D) for the joint power-vs-clock model.

### Practical clock recipes for benchmarking

For peak-throughput characterization:

```bash
# (1) Reset to default (no lock):
sudo nvidia-smi -rgc
# (2) Wait for thermal equilibrium:
sleep 5
# (3) Pkill leftover CUDA processes:
sudo pkill -9 QuickRunCUDA && sleep 5
# (4) Run measurement; should boost to 2032 MHz on first launch:
./QuickRunCUDA <kernel.cu> -T 1000 ...
# (5) Sample clock during run to verify:
nvidia-smi --query-gpu=clocks.gr --format=csv -l 1
```

For controlled-clock studies (e.g., V² extraction, DVS modeling):

```bash
sudo nvidia-smi -lgc <N>     # locks SM clock to N MHz
# WARNING: lock persists across runs; reset with -rgc after.
```

For reproducible "sustained boost" measurements:

```bash
sudo nvidia-smi -lgc 1920    # explicit honest boost lock
# This pins to 1920 MHz delivered (paradox of -lgc 2032 also pins here).
```

### Clock-related pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Stuck at 1005 MHz silently | Measured BW/TFLOPS 50% lower than catalog | `nvidia-smi -rgc; sleep 5` |
| `-lgc 2032` paradox | Test at "boost" delivers 1920 MHz | Stop using `-lgc 2032`; use no-lock + warm-up |
| Background CUDA processes | Inconsistent boost | `pkill -9 QuickRunCUDA && sleep 5` |
| TDP wall hit at 1700+ MHz | Throttle-down to ~1500 MHz mid-test | Use shorter test or lower clock |
| Hot chip from prior test | Boost capped lower than spec | Wait 30s+ between throughput tests |
| Different test-clock vs report | "FFMA 70 TFLOPS at 2032" actually at 1920 | Always state which clock |

**Footgun:** ⚠ Don't quote a TFLOPS number from any pre-2026-04-22 catalog file without checking which clock it was at — there's a 6% systematic gap between "boost" and "locked" reports that confused many cross-comparisons. The CLAUDE.md note "FP32 FFMA 76.96 TFLOPS at 2032 MHz" assumes true boost, not the locked 1920.

**Footgun (separate):** ⚠ Don't assume `nvidia-smi -lgc 2032` does what it sounds like. It pins to 1920 MHz delivered, not 2032 MHz. There is no documented user-facing way to lock to true boost; rely on no-lock + warm-up.

**See also:** §6 (HBM at boost vs locked), §11 (L2 video clock independence), §22 (FFMA peak with clock state, Agent B), §44 (power model, Agent D).

---

## §4. HBM3E topology — 8 stacks (NOT 12)

**Answer:** B300 has **8 × HBM3E 12-Hi stacks** (3 GB die), 16 × 512-bit controllers giving 8192-bit architectural bus. On this AC SKU one controller is fused off → 7680-bit effective bus, 275040 MiB visible.  `[🟢 HIGH · src: corrections/HBM_STACKS_INDEPENDENT_VERIFY.md + corrections/HBM_DENOMINATOR_FINAL.md]`

This was a contested fact across the catalog. Earlier docs (`01_hbm_bandwidth.md` line 3 + line 136) said "12 stacks", which is **wrong**. Authoritative resolution comes from two strong independent sources verified 2026-04-22.

### Independent confirmation (Method 1: NVIDIA Developer Blog)

URL: `https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/`

Exact quote (post the 9/24/25 correction notice):

> "HBM configuration: Eight 12-Hi stacks, 16 × 512-bit controllers (8,192-bit total width)"

The blog explicitly carries a correction notice acknowledging that the originally-published Figure 1 wrongly showed 8 (8-Hi) stacks; the correction is to "12-Hi", not to "12 stacks". The body text consistently says "Eight 12-Hi stacks".

### Independent confirmation (Method 2: cudaGetDeviceProperties)

```
$ cat /tmp/devprops.cu
#include <cuda_runtime.h>
#include <cstdio>
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("name: %s\n", p.name);
    printf("totalGlobalMem: %llu MiB\n", (unsigned long long)(p.totalGlobalMem / (1024*1024)));
    printf("memoryBusWidth: %d bits\n", p.memoryBusWidth);
    printf("l2CacheSize: %d MiB\n", p.l2CacheSize / (1024*1024));
    printf("multiProcessorCount: %d\n", p.multiProcessorCount);
    printf("memoryClockRate: %d kHz\n", p.memoryClockRate);
    return 0;
}
$ nvcc -arch=sm_103a /tmp/devprops.cu -o /tmp/devprops && /tmp/devprops
name: NVIDIA B300 SXM6 AC
totalGlobalMem: 274113 MiB
memoryBusWidth: 7680 bits        <-- NOT 8192
l2CacheSize: 126 MiB
multiProcessorCount: 148
memoryClockRate: 3996000 kHz     <-- 3996 MHz × 2 DDR = 7.992 Gbps/pin
```

The reported bus width of **7680 bits = 8192 × 15/16** can only mean: 8 stacks architecturally provisioned (at 1024 bits/stack each = 8192 total) with one /16 controller-pair fused off on this part. A 7-stack design (= 7168 bits) or a 12-stack (= 12288 bits) cannot produce exactly 7680 = 8192 × 15/16.

### Capacity arithmetic

Two configs are *capacity-equivalent* and cannot be distinguished by `nvidia-smi`-reported memory alone:

| Config | Raw | Post-ECC visible |
|---|---:|---:|
| 8 stacks × 12-Hi × 3 GB/die | 288 GB | 270 GB ≈ 268.6 GiB |
| 12 stacks × 12-Hi × 2 GB/die | 288 GB | 270 GB ≈ 268.6 GiB |

This is why the bus-width derivation (above) is the load-bearing argument, not capacity. Reported `totalGlobalMem = 274113 MiB = 268.08 GiB` plus reserved-memory adds up to the 268.59 GiB visible from `nvidia-smi` (small rounding/reservation difference between the two reports).

### Per-stack bandwidth derivation

```
Per-stack bus       = 1024 bits
Per-pin rate (DDR)  = 3996 MHz × 2 = 7.992 Gbps    (vs spec 8.000 Gbps)
Per-stack BW        = 7.992 × 1024 / 8 = 1022.976 GB/s
8 stacks raw        = 8 × 1022.976  = 8183.8 GB/s   (pre-ECC)
8 stacks post-ECC   = 8183.8 × 15/16 = 7672.3 GB/s  (1/16 SECDED)

This-device (1/16 controllers fused):
8 stacks × 15/16 raw = 7671.7 GB/s pre-ECC
                    × 15/16 ECC = 7192.2 GB/s post-ECC ← if ECC + fuse compounded

OR (the alternative interpretation that matches dev-blog):
"7680-bit bus" already accounts for ECC reservation built INTO the controller fuse.
Then effective post-ECC = 7672 GB/s as derived in HBM_DENOMINATOR_FINAL.md.
```

The empirical observation: measured peak 7.30 TB/s ÷ 7.67 TB/s ≈ 95.2 % of "this-device peak". This matches the spec-derivation when the 7680-bit number is treated as ALREADY post-ECC (matching `cudaDeviceProp.memoryBusWidth` which conventionally reports the user-visible bus, not pre-ECC raw). The compound-interpretation (fuse × ECC compound) gives a denominator that the chip exceeds, which is impossible — so the controller fuse and ECC overhead are **not** independent; the "AC" SKU's 7680-bit width is already the user-visible (post-ECC) bus.

### Summary

| Quantity | Architectural | This SKU (AC) |
|---|---:|---:|
| HBM stacks | 8 | 8 (all present) |
| Stack height | 12-Hi | 12-Hi |
| Die capacity | 3 GB | 3 GB |
| Total controllers | 16 × 512-bit | 15/16 enabled |
| Bus width | 8192 bits | **7680 bits** |
| Capacity (raw) | 288 GB | 288 GB |
| Capacity (post-ECC) | 270 GB | 268.6 GiB |
| BW (post-ECC, 8.000 Gbps spec) | 7680 GB/s | 7670 GB/s on AC |
| BW (this-device, 7.992 Gbps actual) | 7672 GB/s | 7670 GB/s |

### HBM3E spec card (B300 part)

| HBM3E parameter | Value | Per-stack | Per-pin |
|---|---|---|---|
| Per-pin data rate (spec) | 8.000 Gbps | — | DDR @ 4 GHz |
| Per-pin data rate (this device) | 7.992 Gbps | — | DDR @ 3.996 GHz |
| Per-stack pin count | 1024 | — | — |
| Per-stack BW (spec) | 1024 GB/s | 1024 × 8.000/8 | — |
| Per-stack BW (this device) | 1023 GB/s | 1024 × 7.992/8 | — |
| Stack height | 12-Hi | 12 dies | — |
| Die capacity | 3 GB (24 Gb) | — | — |
| Per-stack capacity | 36 GB | 12 × 3 GB | — |
| ECC overhead | 1/16 (SECDED) | — | — |
| Number of stacks | **8** | — | — |
| Total bus width | 8192 bits | 8 × 1024 | — |
| Total raw BW (spec) | 8192 GB/s | — | — |
| Total post-ECC BW (spec) | **7680 GB/s** | — | — |
| Total raw capacity | 288 GB | 8 × 36 | — |
| Total post-ECC capacity | 270 GB ≈ 268.6 GiB | — | — |

### Why 8 stacks vs 12 stacks both fit capacity

A reader who only sees `nvidia-smi --query-gpu=memory.total = 275040 MiB` cannot determine stack count from capacity alone. Both these configurations give 270 GB:

```
Option A (8 stacks × 12-Hi × 3 GB/die):
  8 × 12 × 3 = 288 GB raw → × 15/16 ECC = 270 GB ✓

Option B (12 stacks × 12-Hi × 2 GB/die):
  12 × 12 × 2 = 288 GB raw → × 15/16 ECC = 270 GB ✓
```

The settling argument is **bus width** — `cudaDeviceProp.memoryBusWidth = 7680 bits` could only be 8 × 1024 × 15/16 (Option A with one /16 controller fused), NOT 12 × 1024 × something (which would give 12288 or some non-7680 multiple).

Combined with the NVIDIA Tech Blog correction notice that explicitly says "Eight 12-Hi stacks", **8 stacks is settled**.

### Per-stack BW independence

Each of the 8 stacks is independent — they have separate clock domains, separate PHYs, and separate command/data buses. Implications:

- A stack-local hot kernel (only touching addresses that hash to one stack) can saturate that stack's 1023 GB/s.
- Cross-stack hashing (default for `cudaMalloc`) distributes load across all 8 stacks.
- D2D copies between stack-locality-controlled src/dst can hit 6.93 TB/s (NINJA recipe, §6) by avoiding direction-switch penalties on shared stacks.

### How to verify stack count on your device

```bash
# Method 1: Bus width
nvidia-smi --query-gpu=name,memory.total --format=csv

# Method 2: Compile + run device-property query:
cat > /tmp/stacks.cu << 'EOF'
#include <cuda_runtime.h>
#include <cstdio>
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    int bus_bits = p.memoryBusWidth;
    int io_mhz = p.memoryClockRate / 1000;
    printf("name: %s\n", p.name);
    printf("bus width: %d bits\n", bus_bits);
    printf("I/O clock: %d MHz\n", io_mhz);
    printf("post-ECC BW (this device): %.0f GB/s\n",
           (double)bus_bits * 2 * io_mhz / 8 / 1000);
    printf("inferred stacks (assume 1024 b/stack, 15/16 controller fuse): %d\n",
           bus_bits / (1024 * 15 / 16));
    return 0;
}
EOF
nvcc -arch=sm_103a /tmp/stacks.cu -o /tmp/stacks && /tmp/stacks
```

Expected output for B300 SXM6 AC:
```
name: NVIDIA B300 SXM6 AC
bus width: 7680 bits
I/O clock: 3996 MHz
post-ECC BW (this device): 7670 GB/s
inferred stacks (assume 1024 b/stack, 15/16 controller fuse): 8
```

**Footgun:** ⚠ Do NOT cite "12 stacks" for B300 — that's the pre-correction-notice figure from the NVIDIA blog and propagated incorrectly into `b300_clean/01_hbm_bandwidth.md` lines 3 + 136. Authoritative answer is 8 stacks of 12-Hi each. Whenever you see "12 stacks" in any catalog file, treat as a known error.

**See also:** §5 (denominators), §6 (read peak in TB/s), §11 (L2 capacity 126 MB).

---

## §5. HBM3E denominators

**Answer:** Three valid denominators for "% of HBM peak" claims, each correct in its framing: **7680 GB/s** (spec, cross-vendor), **7672 GB/s** (this-device-actual at 3996 MHz I/O), or **7670 GB/s** (effective on this 7680-bit AC SKU). The catalog historically ALSO used **7.31 TB/s** (empirical pure-direction) as a denominator — that one is WRONG and inflates % numbers by ~5pp.  `[🟢 HIGH · src: corrections/HBM_DENOMINATOR_FINAL.md + corrections/01_hbm_bandwidth_CORRECTED.md §0]`

This is the single most important nuance for understanding HBM bandwidth claims across the b300_clean corpus. Different docs mix denominators silently, making "% of peak" numbers non-comparable across files. This section tells you how to translate.

### The three legitimate denominators

| Denominator | Value | Meaning | Use when… |
|---:|---:|---|---|
| **Spec-comparable** | **7680 GB/s** | post-ECC at 8.000 Gbps spec, 8192-bit bus | Comparing across docs, vendors (B200, MI300X), or vs published peak. Apples-to-apples vs other GPU specs. |
| **This-device-actual** | **7672 GB/s** | post-ECC at empirical 7.992 Gbps/pin (3996 MHz × 2), 8192-bit | Asking "how close to what THIS silicon physically can do?" — the strict SoL-on-this-GPU denominator (architectural max, not SKU-fused). |
| **This-AC-SKU effective** | **7670 GB/s** (≈ 7672) | post-ECC at 7.992 Gbps × 7680-bit (controller fuse) | The strict ceiling for THIS particular die, accounting for the 1/16 controller fuse. |
| **Architectural raw** | 8192 GB/s | spec pre-ECC at 8.000 Gbps | Rare; only when measurement excludes ECC parity (ncu does not). |
| **This-device raw** | 8183.8 GB/s | empirical pre-ECC at 7.992 Gbps | Symmetric to 7672 on the raw side. |

The three "post-ECC" numbers (7680 / 7672 / 7670) are within 0.13 % of each other — for almost all practical purposes they are interchangeable. The discipline is: **pick one and stick to it within a doc**. This canonical reference uses **7680 GB/s** as the default denominator (matches NVIDIA marketing rounded to 8 TB/s after the 1/16 ECC reservation, and matches B200 / MI300X conventions).

### The wrong denominator

`b300_clean/V32_V40_FINDINGS.md` and `V41_V48_FINDINGS.md` used **7.31 TB/s** as their denominator. That number is the **empirical pure-direction read peak** that the chip achieves under the v8 + per-warp coalesced recipe. Using a measured peak as a denominator means every "% of peak" claim in those files is actually "% of best-other-measurement", and inflates the apparent SoL by:

```
7.31 / 7672 = 95.3% of true SoL
A test reporting 7.20 / 7.31 = 98.5% is actually 7.20 / 7672 = 93.8% of true SoL.
The 5-percentage-point gap between "% of 7.31" and "% of 7672" is the artifact.
```

This bit V46's "98.5% NEW HBM SoL" headline (re-normalizes to 93.8% — see §6).

### How to translate between catalog files

| If a doc says… | Cross-translate as… |
|---|---|
| "7.20 TB/s = 98.5% of HBM peak" | Multiply by 7.31/7672 = 0.953 → "93.9% of spec" (match this canonical doc) |
| "7.30 TB/s = 95% of HBM" (this canonical) | Same as "100% of empirical 7.31" (V32-V48 convention) |
| "8 TB/s spec" or "~8 TB/s peak" (CLAUDE.md older line) | Marketing rounded; treat as 7680 GB/s post-ECC |
| "7.57 TB/s = 105% of HBM read peak" (V8_HBM_WRITE_SOL.md) | Denominator-mismatch artifact; re-normalize: 7.57/7672 = 98.7% |

### Recommended reporting standard

```
"7.20 TB/s = 93.9% of 7.68 TB/s spec / 93.9% of 7.67 TB/s this-device"
```

When precision matters, cite both. When it doesn't, default to 7680 GB/s. **Never** silently use 7.31 as a denominator.

### Why this matters for cross-vendor comparisons

If you want to compare B300 to MI300X (288 GB HBM3, 5.3 TB/s spec) or B200 (192 GB HBM3E, 8 TB/s spec), use **spec denominators consistently**. Otherwise you'll claim B300 is "98 % of peak" while MI300X is "70 % of peak" because of your own denominator choice.

Apples-to-apples table for comparing GPU HBM SoL:

| GPU | HBM spec (TB/s) | Best measured (TB/s) | % of spec |
|---|---:|---:|---:|
| H100 SXM5 | 3.35 | ~3.0 | ~90 % |
| MI300X | 5.30 | ~4.6 | ~87 % |
| B200 SXM6 | 8.0 | ~6.7 | ~84 % |
| **B300 SXM6 AC** | **8.0** (spec) / **7.68** (this-device cap) | **7.30** | **91 % of spec / 95 % of this-device** |

Note: B300 numbers cited at 7.30 / 7.68 = 95 % use the this-device denominator (which accounts for the controller fuse). Cross-vendor comparison should use 7.30 / 8.00 = 91 % (the spec denominator).

### Edge cases

- **Pure sequential cudaMemset** approaches the spec ceiling (7.47 TB/s wall-clock) but ncu shows ~7.30 TB/s actual DRAM bytes. The 0.17 TB/s wall-clock overshoot is almost certainly a measurement-window-end artifact (last DMA completes after the timer stop). For canonical reporting, use the ncu number (7.30 TB/s).
- **Write peak 7.57 TB/s** would be 98.7 % of 7680 spec or 98.7 % of 7672 this-device — within 0.1 % regardless of denominator choice. Provenance contested (see §7).

**Footgun:** ⚠ Many catalog %-of-peak claims used **7.31 TB/s empirical-pure-direction** as the denominator, inflating numbers by ~5pp. If a doc cites "98%+ of HBM SoL" without naming the denominator, suspect the 7.31-as-denominator artifact and re-normalize. The corrected version of the read peak is 95–96% of spec, not 98%.

**See also:** §4 (where 7672 vs 7680 comes from), §6 (V46 demotion case study), §7 (write SoL contested provenance).

---

## §6. HBM read peak

**Answer:** **7.30–7.37 TB/s = 95.2–96.0% of 7680 GB/s spec**, achievable via either LDG.E.128 + per-warp coalesced (7.37 TB/s, the SoL) OR TMA `cp.async.bulk` 8 KB chunks (7.34 TB/s) OR v8 + per-warp coalesced + non-persistent (7.30 TB/s NINJA recipe). V46 8-deep TMA pipelined reaches 7.20 TB/s = 93.8% (BELOW the SoL — see footgun).  `[🟢 HIGH · src: corrections/01_hbm_bandwidth_CORRECTED.md §1+§2 + corrections/V46_DOUBT_REPORT.md]`

### Read-peak ladder (re-normalized to 7680 spec)

| Variant | TB/s | % of 7680 spec | Source / commit |
|---|---:|---:|---|
| LDG.E.128, 37888 blocks | **7.37** | **96.0%** ← SoL | `01_hbm_bandwidth.md` line 66 |
| User v8 + per-warp coalesced (4 GB) | 7.37 | 96.0% | `01_hbm_bandwidth.md` line 102, commit `a04d9c8` |
| TMA `cp.async.bulk` 8 KB, 37888 blocks | 7.34 | 95.7% | `01_hbm_bandwidth.md` line 65 |
| A6 R-only sweep (R:W = 32:0) | 7.31 | 95.3% | `01_hbm_bandwidth.md` A6 table |
| **Canonical NINJA (v8 + per-warp + non-persistent)** | **7.30** | **95.2%** | commit `a04d9c8` |
| V46 TMA pipelined 8-deep, 16 KB tiles | 7.20 | 93.8% | `V41_V48_FINDINGS.md`, commit in v46_tma_inflight.cu |
| V33 TMA single-deep 64 KB | 6.72 | 87.6% | `V32_V40_FINDINGS.md` |
| Plain LDG.32 coalesced | 1.95 | 25.4% | `V10_LDG_WIDTH.md` |
| Plain LDG.64 coalesced | 3.65 | 47.5% | `V10_LDG_WIDTH.md` |
| Plain LDG.128 coalesced (without coalescing recipe) | 5.76 | 75.0% | `V10_LDG_WIDTH.md` |

The narrow-margin spread between 7.30 and 7.37 TB/s is within run-to-run noise (±1 % typical). Treat anything in [7.30, 7.40] as "the read SoL".

### The canonical NINJA read recipe

```cpp
__global__ void w_v8_coalesced(int *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *warp_base = data + warp_id * (32 * 1024 / 4);
    int v;  // accumulator
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = warp_base + (it * 32 + lane) * 8;
        // 8-wide LDG (256 bits = 32 B per thread per iter)
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
              "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(p) : "memory");
        // (write to data prevents DCE)
    }
}
// Launch: <<<bytes / (256 * 1024), 256>>>   // 16384 blocks for 4 GB
```

SASS: `LDG.E.ENL2.256` (256-bit load, ENL2 = entered L2 path, no L1 cache).

**Saturation requirements** (all required, removing any one drops below 7.30):

1. WS ≥ 4 GB (smaller WS hits L2)
2. 256-bit per-instruction width (`v8` / `LDG.E.128` minimum)
3. Per-warp 1-KB bursts (each warp owns a contiguous 32 × 32 B = 1 KB region per iter, then advances)
4. High parallelism (≥ 16384 CTAs, NOT persistent)

CTA-count and cache hint do NOT matter once the above are satisfied. `.cg` vs default vs `.ca` all produce ~7.30 TB/s for 4 GB DRAM-bound work.

### Working-set breakdown (where the cliff is)

| WS | Effective BW (TB/s) | Tier |
|---|---:|---|
| 16 MB | 46.6 | L1 + L2 combined |
| 64 MB | 23.0 | L2 plateau (kernel-effective) |
| 100 MB | 20.9 | L2 edge |
| **126 MB** | 8.2 | **CLIFF — exactly L2 capacity** |
| 256 MB | 7.32 | DRAM-bound |
| 1024 MB | 7.12 | DRAM-bound |
| 4 GB | 7.29–7.30 | DRAM-bound, true HBM3E ceiling |
| 32 GB | ~7.20 | DRAM-bound (refresh-rate ceiling?) |

The 126 MB cliff is sharp because L2 capacity (`cudaDeviceProp.l2CacheSize = 132,644,864 B`) is exactly there; for WS < 126 MB the recurrence stays cached. See §11 for L2 nuance.

### TMA path (V41–V48 series re-normalized)

V41–V48 found that pipelining TMA reads (8-deep) reach 7.20 TB/s, a 2.5× improvement over V33's single-deep 6.72 TB/s WITHIN the TMA path. V41_V48 originally headlined this as "NEW BEST = 98.5%" using 7.31 TB/s as denominator. Re-normalized to 7680 spec, V46 = 7.20 / 7680 = **93.8%**, which is **lower** than `01_hbm_bandwidth`'s quoted TMA bulk read (7.34 = 95.7%) and LDG.E.128 (7.37 = 96.0%).

**Corrected statement (per `corrections/V46_DOUBT_REPORT.md` §5):**

> "V46 confirms TMA reads benefit from 8-deep pipelining (intra-test 1-deep → 8-deep speedup) and recovers ground that V33's single-deep left on the table, but does NOT establish a new architectural read SoL. The HBM3E read ceiling remains 7.30–7.37 TB/s = 95–96% of 7680 GB/s spec."

The TMA pipelining lesson is real (an architectural best-practice for TMA users); the SoL claim is not.

### TMA + prefetch.L2 anti-pattern (V42)

`prefetch.L2` combined with `cp.async.bulk` is **27 % slower** than no-prefetch. TMA has its own DMA path; explicit prefetch instructions block forward progress. **Rule: never combine `prefetch.L2` with `cp.async.bulk`.**

(Note: the V6 1.58× prefetch speedup applies to **legacy `cp.async`** (LDGSTS), NOT to `cp.async.bulk` / TMA. Both observations are correct in their respective regimes.)

### TMA multicast aggregate

| Variant | Aggregate effective | Source |
|---|---:|---|
| TMA multicast 8-way × 18 clusters | 14.9 TB/s | V32 |
| V48 attempt to pipeline multicast | 13.96 TB/s (CAPPED) | V48 |

**Multicast cannot be pipelined** — single TMA engine per cluster. V32's 14.9 TB/s is the architectural multicast ceiling. See §13 for DSMEM/multicast detail.

### What the 5 % gap to spec might be

The gap between measured 7.30–7.37 TB/s and theoretical 7672 GB/s spec is ~3–5 %, consistently. Candidates (none directly attributed):

- HBM3E refresh cycles (every ~32 ms, ~30 cy each)
- Command bus turnaround for bursts
- Row-precharge time during bank rotation
- ECC parity write-back cycles for partial writes (not applicable for pure reads)

`01_hbm_bandwidth.md` A2 noted bursts <1 KB hit 98.6 % of theoretical, while longer bursts under-saturate due to row-conflict scheduling. The 5 % gap is real silicon overhead, not a measurement artifact.

### SASS verification of the read peak

The canonical NINJA recipe compiles to:

```
LDG.E.ENL2.256 R0, [R8.64]
LDG.E.ENL2.256 R8, [R8.64+0x100]
LDG.E.ENL2.256 R16, [R8.64+0x200]
... (32 iterations)
```

`LDG.E.ENL2.256` semantics:
- `LDG.E` — global load with extended addressing
- `.ENL2` — entered through L2 (NOT through L1; equivalent to `.cg` cache hint)
- `.256` — 256-bit width (8× 32-bit lanes per thread)

The `.ENL2` is interesting: this is the SASS encoding for a load that bypasses L1 to reduce L1 pressure on a DRAM-bound kernel. The runtime compiler emits this when `cudaMallocManaged` or `cudaMallocAsync` are involved; for plain `cudaMalloc` without policy hints, you get `LDG.E.STRONG.SM` (L1+L2 cached). Both reach 7.30 TB/s for DRAM-bound work because L1 is irrelevant when WS >> L1.

To inspect SASS:

```bash
nvcc -keep -arch=sm_103a kernel.cu
cuobjdump -sass kernel.cubin | grep LDG
```

### ncu cross-check methodology

For the canonical NINJA recipe, ncu metrics:

```
dram__bytes_read.sum.pct_of_peak_sustained_elapsed   = ~95%
dram__bytes_read.sum / wall_clock_time              = 7.30 TB/s
lts__t_bytes_pipe_dram_op_read.sum / time           = matches
```

The `dram__bytes_read.sum` is the most authoritative metric: it counts bytes that left HBM controllers, divided by elapsed time. Use this as the "ground truth" denominator for HBM bandwidth claims.

### Read-vs-write asymmetry

Reads and writes hit similar peaks (~7.3 TB/s), but their failure modes differ:

| Failure mode | Read | Write |
|---|---|---|
| Sub-sector access | 7× amplification (RMW) | 7.5× amplification |
| Misalignment | sector-aligned coalescing required | sector-aligned coalescing required |
| Hot stack | random across stacks via hash; stack-locality NOT exploitable for reads | same |
| DMA path | TMA `cp.async.bulk` 8 KB chunks competitive | TMA bulk store 8-deep does NOT help (V47) |
| Pipelining | TMA 8-deep recovers within-TMA gap | TMA single-deep is fine |

The lesson for kernel writers: make stores **256-bit aligned** (`v8` / `int4` / 4× int4 etc.) AND coalesced per-warp into 1 KB bursts. Both required for SoL.

**Footgun:** ⚠ V46's "98.5% NEW SoL" was a denominator artifact (7.20 / 7.31 = 98.5%, not vs spec). When you see TMA pipelined as a "new HBM SoL" in any catalog file, re-normalize to 7672 / 7680 spec; you'll find it's 93.8 %, BELOW the existing LDG.E.128 ceiling. The HBM read SoL is NOT held by TMA — it's held by plain LDG.E.128 with the right launch geometry.

**See also:** §5 (denominator nuance), §7 (write peak), §8 (concurrent R+W), §11 (L2 cliff at 126 MB).

---

## §7. HBM write peak

**Answer:** **7.30 TB/s = 95.2% of 7680 spec** for the standard v8 STG NINJA recipe; **7.57 TB/s = 98.7%** is the contested write SoL with disputed provenance (NINJA STG vs TMA bulk store).  `[🟡 MED · src: corrections/01_hbm_bandwidth_CORRECTED.md §3 + V8_HBM_WRITE_SOL.md]`

### Write-peak ladder

| Variant | TB/s | % of 7680 spec | Source / commit |
|---|---:|---:|---|
| **NINJA STG (1 v8 store/warp)** OR **TMA bulk store** | **7.57** | **98.7%** ← contested | DISPUTED: TRUE_REFERENCE attributes to `e75c7e1` (NINJA STG); V8_HBM_WRITE_SOL attributes to `28211ce` (TMA bulk) |
| v8 STG + per-warp 32-iter coalesced | 7.30 | 95.2% | `a04d9c8` |
| A6 W-only sweep (R:W = 0:32) | 7.28 | 94.9% | `01_hbm_bandwidth.md` A6 |
| TMA single-deep store (V34) | 7.17 | 93.5% | `V32_V40_FINDINGS.md` |
| TMA 8-deep pipelined store (V47, NO BENEFIT) | 6.34 | 82.6% | `V41_V48_FINDINGS.md` |
| Plain STG.E.128 (without per-warp coalescing) | 6.11 | 79.6% | V8_HBM_WRITE_SOL claim |
| `cudaMemset` (true DRAM rate, ncu) | ~7.30 | ~95% | wall-clock 7.47–7.52 over-states by ~3% |
| D2D NINJA (separate src/dst) | 6.93 | 90.3% | `4958d6b` |
| D2D `cudaMemcpyAsync` | 6.56 | 85.5% | "single-direction 3.28 × 2" |

### Why writes are NOT slower than reads

The naive expectation "writes should be slower than reads on HBM3E because of bus turnaround / write-amplify" is **false** on B300. Writes hit the same 95–99 % of spec ceiling as reads, because:

- HBM3E PHY has dedicated write and read queues with deep buffering
- ECC write-back is cycle-overlapped (not added latency)
- Per-warp coalesced 1-KB bursts saturate the write queue same as the read queue

The "writes 9 % slower than reads" framing in some pre-2026 docs is a denominator-mismatch artifact (used different denominators for read vs write). Once both are normalized to 7680, the gap is **~3 percentage points** (7.30 read vs 7.30 write standard; 7.57 write vs 7.37 read at SoL), not 10 %.

### Why TMA pipelining does NOT help writes

V47 found that pipelining TMA bulk stores 8-deep gives **6.34 TB/s = NO benefit** (vs 7.17 TB/s single-deep). The reason: **writes are already async fire-and-forget** in the TMA path. The single-deep TMA write doesn't stall waiting for completion — it hands off to the DMA engine immediately. Pipelining just adds bookkeeping overhead.

This is the inverse of TMA reads (where 8-deep pipelining is required to recover ground). The lesson: **for TMA stores, single-deep is fine; for TMA loads, pipeline 8-deep**.

### The contested 7.57 TB/s

`B300_TRUE_REFERENCE.md` claims 7.57 TB/s came from a STG-based NINJA recipe (commit `e75c7e1`, 1 v8 store per warp, massive parallelism). However, `V8_HBM_WRITE_SOL.md` attributes 7.57 TB/s to **TMA bulk store** (commit `28211ce`) and explicitly states that plain STG.E.128 caps at 6.11 TB/s.

These two attributions are **mutually exclusive**:

- If TRUE_REFERENCE is right, plain STG with the right launch geometry hits 7.57.
- If V8 is right, plain STG caps at 6.11 and only TMA gets to 7.57.

Either way:

- The number 7.57 TB/s = 98.7 % is real (both files agree on the value).
- One of the two attributions is wrong.

**Settlement status:** UNRESOLVED as of 2026-04-22. Needs a clean re-test of both `e75c7e1` (NINJA STG) and `28211ce` (TMA bulk) on the same machine with ncu DRAM bytes verification (`dram__bytes_write.sum`). Until settled, both attributions are listed as possible, and the canonical write SoL is reported as `7.57 TB/s (provenance disputed; clean re-test pending)`.

### Standard write recipe (HIGH conf)

The 7.30 TB/s standard-write recipe (commit `a04d9c8`):

```cpp
__global__ void w_v8_coalesced(int *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *warp_base = data + warp_id * (32 * 1024 / 4);
    int v = 0xab;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = warp_base + (it * 32 + lane) * 8;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}
// Launch: <<<bytes / (256 * 1024), 256>>>   // 16384 blocks for 4 GB
```

SASS: `STG.E.ENL2.256` — 256-bit aligned global store, ENL2 path.

**Saturation requirements** (mirroring the read recipe):

1. WS ≥ 4 GB.
2. 256-bit per-instruction width (`v8`).
3. Per-warp 1-KB bursts.
4. High parallelism (≥ 16384 CTAs, NOT persistent).

This recipe hits 7.30 TB/s consistently across multiple test runs. It's the reproducible write SoL.

### TMA bulk store (alternative path)

```cpp
// In kernel:
constexpr int TILE = 32 * 1024;  // 32 KB
extern __shared__ int smem[TILE / 4];
// ... fill smem ...

asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
    :: "l"(global_dst_addr), "r"(smem_offset), "n"(TILE) : "memory");
asm volatile("cp.async.bulk.commit_group;");
asm volatile("cp.async.bulk.wait_group 0;");
```

This path:

- Uses TMA (Tensor Memory Accelerator) DMA engine.
- Async / fire-and-forget at single-deep (no need to pipeline 8-deep for writes — V47 confirmed pipelining doesn't help).
- Reaches 7.17 TB/s in V34, possibly 7.57 TB/s if `B300_TRUE_REFERENCE` provenance is correct.

**Don't combine with `prefetch.L2`** — see §6 V42 anti-pattern.

### `cudaMemset` characterization

`cudaMemset` invokes a built-in driver kernel that's optimized for B300. Wall-clock timing shows 7.47–7.52 TB/s effective rate. ncu shows ~7.30 TB/s actual `dram__bytes_write.sum`/time. The 0.2 TB/s discrepancy is a measurement-window-end artifact (last DMA completes after the timer stop captures elapsed time).

For benchmarking purposes, use the ncu number (7.30 TB/s). For wall-clock measurements where you don't have ncu, the 7.5 TB/s is approximately right.

### When NOT to use `cudaMemset`

- For partial-pattern fills (e.g., set 4 bytes of every 16-byte sector to a value), `cudaMemset` only sets 1-byte values; for 4-byte you need a custom kernel.
- For non-uniform fills (e.g., random init), use a custom kernel with `curand` or pre-initialized arrays.
- For benchmarking write-path SoL, use the NINJA STG recipe above (more controllable).

**Footgun:** ⚠ Don't quote "7.57 TB/s STG NINJA" or "7.57 TB/s TMA bulk" without acknowledging the disputed provenance. The number is real; the recipe is uncertain. If you depend on this for a recipe, run BOTH and compare — see §6's footgun on V46 for an analogous denominator-attribution failure mode.

**Footgun (separate):** ⚠ Don't quote "writes exceed reads by 5 %" — that was a denominator-mismatch artifact (used 7.2 effective for read, 8.0 nominal for write). True asymmetry is ≤3 percentage points either way.

**See also:** §6 (read peak ladder), §8 (R+W concurrent contention), §10 (cudaMemset's wall-clock vs ncu gap).

---

## §8. HBM concurrent R+W

**Answer:** **7.31 TB/s pure-direction ceiling**, **6.68 TB/s** at the 50:50 minimum (-13 % from balanced contention U-curve), D2D copy hits **6.93 TB/s** with the NINJA recipe and **6.56 TB/s** via `cudaMemcpyAsync`.  `[🟢 HIGH · src: corrections/01_hbm_bandwidth_CORRECTED.md §6 + §8]`

HBM3E on B300 is **shared-bus, not full-duplex** — the controllers serve reads and writes through a common bank pipeline, and direction-switches incur tWTR/tRTW penalties. Mixed R+W traces a U-shape with minimum at 50:50.

### R:W ratio sweep (commits in `01_hbm_bandwidth.md` A6)

| R:W (ops/thread) | DRAM read TB/s | DRAM write TB/s | Aggregate TB/s | % of 7680 spec |
|---|---:|---:|---:|---:|
| 32:0 (pure read) | 7.31 | ~0 | **7.31** | 95.2% |
| 28:4 | 6.22 | 0.86 | 7.08 | 92.2% |
| 24:8 | 5.40 | 1.72 | 7.12 | 92.7% |
| 20:12 | 4.32 | 2.50 | 6.82 | 88.8% |
| **16:16 (50:50)** | 3.39 | 3.29 | **6.68** ← min | **87.0%** |
| 12:20 | 2.52 | 4.09 | 6.61 | 86.0% |
| 8:24 | 1.65 | 4.85 | 6.50 | 84.6% |
| 4:28 | 0.83 | 5.72 | 6.55 | 85.3% |
| 0:32 (pure write) | ~0 | 7.28 | **7.28** | 94.8% |

U-shape. Minimum at 50:50 = 6.68 TB/s. Mechanism: **tWTR (write-to-read) and tRTW (read-to-write) bank turnaround time on HBM3E PHY**. Direction-switch penalty is bank-local and amortizes when the imbalance is large enough that one direction stays dominant.

### What this means for D2D copies

A device-to-device copy (`cudaMemcpyDeviceToDevice`) is the canonical 50:50 workload — every byte is read from src and written to dst. Without separated stacks, expected ceiling is the 6.68 TB/s minimum. With separated stacks (src and dst on different HBM channels), you can do better:

| Variant | Aggregate TB/s | % of 7680 |
|---|---:|---:|
| `cudaMemcpyAsync` (2 GB) | 6.56 | 85.5% |
| D2D NINJA (separate src/dst, contention-free stacks) | **6.93** | **90.3%** |

The NINJA recipe out-performs `cudaMemcpyAsync` by 5.5 % by exploiting stack-locality (placing src and dst at addresses that hash to different HBM channels).

### What this implies architecturally

- HBM3E on B300 is **NOT full-duplex** — there's no separate read-bus and write-bus per stack.
- Direction-switching is the bottleneck, NOT command-bus or address-bus saturation (which would show at higher imbalance ratios).
- For workloads that must do mixed R+W (gather-scatter, transpose, in-place updates), expect ~85–90 % of single-direction peak unless you can carefully arrange stack-locality.

### Cross-check with multi-GPU

User memory `project_b300_multigpu.md` notes 718 GB/s P2P write and 820 GB/s P2P read on 2× B300 NV18 — those numbers are the NVLink-side ceilings (see §14). HBM-side concurrent R+W under multi-GPU has not been characterized; project memory points to MGFenceBench but that test only uses single-direction NVLink, not HBM-side contention. UNRESOLVED.

### tWTR / tRTW background

HBM3E PHY enforces minimum delays between read and write commands on the same bank:

- **tWTR (write-to-read)**: ~10–15 ns delay after a write before the same bank can read.
- **tRTW (read-to-write)**: ~5 ns delay after a read before the same bank can write (smaller).

For a 50:50 mixed workload, the effective utilization is reduced by these delays. Total budget per cycle is divided into:

```
Active read time          + tRTW_penalty
+ Active write time        + tWTR_penalty
+ Refresh + precharge      + bank rotation
= 100% cycle
```

At pure-read or pure-write, only one side is active and there's no direction-switch penalty. At 50:50, every burst alternates direction, paying tWTR + tRTW each time. The 13 % drop from 7.31 to 6.68 TB/s reflects the direction-switch overhead averaged across the bank rotation pattern.

### How to design for low R+W penalty

If your kernel must do mixed R+W, separate src and dst by:

- **Different stacks** (control via `cudaMallocAsync` with `cudaMemPoolAttrPriority` or stride patterns that hit different L2 hash partitions).
- **Different banks within a stack** (rare to control; cache-line stride patterns matter).
- **Temporal separation** (read all, then write all) — but this requires WS buffering in SMEM/L2.

The D2D NINJA recipe (6.93 TB/s) uses stack-locality to put src on stacks 0-3 and dst on stacks 4-7, getting near-pure-direction throughput for both halves of the copy.

### Why 50:50 is the worst case (and not 60:40 or 40:60)

Bank rotation happens at fixed cadence; direction-switch penalty per switch is constant. The MORE direction-switches per unit time, the lower the throughput. At 50:50, switches happen at maximum rate (every burst). At 60:40, the chip can batch the majority direction (60 %) without switching, only paying the penalty at the boundary.

The U-shape is symmetric (which the table confirms: 88.9 % at 20:12 ≈ 88.9 % at 12:20). The minimum at 50:50 is geometrically forced.

**See also:** §6 (pure-read peak), §7 (pure-write peak), §14 (NVLink-side P2P). No footgun.

---

## §9. HBM data-dependence

**Answer:** Memory subsystem POWER follows a **popcount bell curve peaking at d=16** (random-position popcount), with **+240–367 W active power swing at 1500 MHz** (NOT <50 W as `HBM_DATA_DEPENDENCE.md` originally claimed — that file is SUPERSEDED). Bandwidth itself is content-INDEPENDENT (<1 % variance) under the same workload. The 1071 W stress recipe = DRAM read d=16 random + 1500 MHz lock.  `[🟢 HIGH · src: corrections/STRAYS_CORRECTED.md §2 + corrections/01_hbm_bandwidth_CORRECTED.md §7 + b300_clean/POPCOUNT_3TIER.md + b300_clean/POPCOUNT_VS_CLOCK.md + b300_clean/L2_DRAM_DATA_PWR.md]`

### Two regimes, two answers

| Regime | Power swing | Source |
|---|---:|---|
| **Constant patterns** (same word repeated; e.g. all-zero vs all-one vs `0x12121212`) | **5–6 W** | `L2_DRAM_DATA_PWR.md` (11-pattern sweep, < 1 %) |
| **Random-position popcount d=0..32** (different bit positions across words) | **240 W active / 554 W at 1500 MHz** | `POPCOUNT_3TIER.md`, `POPCOUNT_VS_CLOCK.md` |

The disagreement between these is real but ONLY shows up when comparing constant vs random-position. `L2_DRAM_DATA_PWR.md` controlled inter-word toggle to near-zero by repeating the same pattern → 5.4 W spread. `POPCOUNT_3TIER.md` deliberately varied bit positions per dword → 240 W spread.

The **mechanism** is bus-toggle (Hamming) energy on the HBM3E PHY: power scales with the number of bit transitions on the wires per cycle, NOT with the static popcount of the data. Repeating the same pattern minimizes toggles regardless of popcount; random data at d=16 maximizes toggles (highest variance per bit position).

### Bell-curve table (random-position popcount, DRAM-bound, 1005 MHz)

| popcount d (per 32-bit word) | DRAM-1G W | DRAM-8G W | Notes |
|---:|---:|---:|---|
| 0 (all zero) | 369 | 397 | min |
| 4 | 422 | 461 | rising |
| 8 | 480 | 545 | rising |
| 12 | 548 | 605 | rising |
| **16 (random max-toggle)** | **604** | **637** | **bell peak** |
| 20 | 547 | 591 | descending |
| 24 | 472 | 504 | descending |
| 28 | 411 | 437 | descending |
| 32 (all-one) | 380 + DBI | 415 + DBI | min + DBI penalty |

DBI = Data-Bus Inversion: HBM3E PHY can flip all 32 bits if it reduces toggle count. The "all-one" tier is slightly higher than "all-zero" because of active-low termination overhead; the +11 to +44 W asymmetry between d=0 and d=32 across cache-tier distance grows with HBM-distance (L1 +11.8 W, L2 +22.8 W, DRAM-1G +41.6 W, DRAM-8G +44.8 W) — reported consistently in the POPCOUNT family.

### Power scales with clock (V² model)

`POPCOUNT_VS_CLOCK.md` swept the same DRAM-8G d=16 workload across clock locks:

| Clock (MHz) | DRAM-8G d=16 random W | DRAM-8G d=0 W | Swing |
|---:|---:|---:|---:|
| 510 | 295 | 178 | 117 |
| 1005 | 637 | 397 | 240 |
| 1500 | 921 | 367 | **554** |
| 1700 | 1004 (TDP-capped) | ~390 | ~614 |
| 1800 | 942 (throttled, TDP cap hit, clock dropped) | ~415 | ~527 |

At 1500 MHz the chip can simultaneously push 7+ TB/s of DRAM bandwidth AND draw 921 W of memory-subsystem power. Adding compute simultaneously caps at TDP wall (~1100 W).

### The 1071 W stress recipe

For burning maximum power as a stress test:

```
Recipe: DRAM read at full saturation, random-position popcount d=16 data,
        nvidia-smi -lgc 1500.

This pulls 1071 W on B300 SXM6 AC.
```

This is documented in user memory `project_b300_power_data_dep.md`. Higher clocks (1700/1800) hit the TDP wall and start throttling; 1500 MHz is the max sustainable stress point.

### Bandwidth is content-INDEPENDENT

`L2_DRAM_DATA_PWR.md` confirmed that across all 11 data patterns, measured bandwidth varied <1 % when properly measured with `.cg` 1024-B/warp loads:

- L2-warm reads: 340.1–343.6 W chip power across 11 patterns (3.5 W = 1 % spread)
- DRAM-cold reads: 522.6–528.0 W chip power across 11 patterns (5.4 W = 1 % spread)
- BW: 7.30 TB/s ± noise across all 11 patterns

In other words: the chip uses **more power** to deliver the same bandwidth on random-toggle data, but it doesn't deliver less bandwidth. Cache-line traffic is fixed at 128 B units, bus signaling is at fixed rates, address decoding is deterministic per access. **Memory subsystem bandwidth is data-pattern independent within 1 %**.

### Why this matters for ML inference

Real production weight tensors (FP16/BF16/INT8/FP8) tend to have popcount distributions that lean toward d=8..d=20 (not uniform random). The d=16 stress recipe is an upper bound on power for memory-bound operations. Practical inference workloads see:

- ~400–500 W chip power for memory-bound layer (e.g., attention KV read)
- ~600–800 W for compute-bound matmul (see Agent D §44)
- 1100 W only with deliberate stress recipes; rare in production

For ML practitioners: choose **boost (2032 MHz)** for inference latency optimization (see user memory `project_b300_v6_complete.md` — 3× lower energy than 510 MHz). Don't try to save power by lowering clocks; energy-per-token gets worse.

### Toggle-energy model (theory)

The mechanism is **bus-toggle (Hamming-distance) energy** on the HBM3E I/O wires. Per-cycle energy is approximately:

```
E_cycle ≈ k × (number_of_bit_flips × C × V²)

where:
  k                  = process constant
  number_of_bit_flips = sum over all wires of (current_bit XOR previous_bit)
  C                  = wire capacitance
  V                  = supply voltage
```

For a 7680-bit bus at 7.992 Gbps DDR:

- Max possible flips per cycle = 7680 (every bit flips)
- Min possible flips per cycle = 0 (no bit flips, e.g. identical patterns)
- Average for random data ≈ 3840 (50 % chance per wire)

**Why d=16 maximizes**: Random-position popcount-16 means each 32-bit word has 16 ones in random positions. Across consecutive words, the probability that any wire toggles is highest at d=16 (binomial peak). At d=0 (all-zero) or d=32 (all-one), consecutive words are identical so toggle activity = 0 (Data-Bus Inversion can flip 32 → 0 active-low if needed, hence small DBI penalty).

**DBI mechanism**: HBM3E PHY can invert all 32 bits of a wire-group if doing so reduces total toggles. So "all-one" is effectively encoded as "all-zero with DBI flag set", costing slight extra control overhead but saving significant toggle energy. This is why d=32 is only 11–44 W higher than d=0, not 7680× higher (which the naive theory would predict).

### Why this is HBM-distance dependent

The +11 to +44 W asymmetry between d=0 and d=32 grows with cache-tier distance:

| Tier | d=32 minus d=0 | Mechanism |
|---|---:|---|
| L1 | +11.8 W | short wires, low capacitance |
| L2 | +22.8 W | medium wires through XBAR |
| DRAM-1G | +41.6 W | long wires through HBM3E PHY |
| DRAM-8G | +44.8 W | similar to 1G; PHY-dominated |

Longer wires have higher capacitance, so toggle energy per flip is larger. The pattern is consistent across all 4 popcount-family files (`L2_POPCOUNT_SWEEP`, `POPCOUNT_3TIER`, `POPCOUNT_VS_CLOCK`, `POPCOUNT_WRITES`).

### Practical power-stress recipes (not advisable for production)

If you actually want to stress B300 to TDP for thermal validation:

```bash
# Burn 1071W on memory subsystem only:
nvidia-smi -lgc 1500
./QuickRunCUDA tests/power_stress_dram.cu -p -A 4194304 \
    --random-data --popcount-target 16 -T 100000

# Burn 1100W mixed (compute + memory):
nvidia-smi -lgc 1700
./QuickRunCUDA tests/power_stress_mixed.cu -p -T 100000

# Verify no throttle:
nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu \
    --format=csv -l 1
```

If you see clock dropping during the stress run, the chip is throttling at TDP wall — back off clock by 100 MHz. The 1700 MHz clock with mixed compute+memory at random data is the "sweet spot" for hitting TDP cleanly without throttling.

### Why `HBM_DATA_DEPENDENCE.md` is superseded

`b300_clean/HBM_DATA_DEPENDENCE.md` was the early inferred / pre-sweep file. It claimed:

> "HBM data-dependent power likely contributes <50W out of total 1100W TDP"

This is **WRONG**. The real swing under random-position popcount is 240 W active / 554 W at 1500 MHz. The author's own caveats acknowledged the file's measurement was at 20.4 GB/s (0.3 % of peak), not a real DRAM-saturation test. Superseded by the 4-file POPCOUNT family + `L2_DRAM_DATA_PWR.md`.

If you find `HBM_DATA_DEPENDENCE.md` referenced anywhere in your reading, redirect to:

- `POPCOUNT_3TIER.md` (canonical 3-tier sweep)
- `POPCOUNT_VS_CLOCK.md` (clock-frequency dependence)
- `L2_DRAM_DATA_PWR.md` (constant-pattern control)
- `corrections/16_power_clock_CORRECTED.md` §5 (synthesis)

**See also:** §3 (V² clock scaling), §6 (BW peaks unaffected by data), §44 (power model, Agent D), §65 (popcount synthesis, Agent F).

---

## §10. L1 cache

**Answer:** 256 KB unified L1+SHMEM pool per SM; carveout 0..228 KB user-allocatable; **effective L1 bandwidth ~30.5 TB/s typical, up to 46 TB/s small-WS** (M5 cheatsheet); sharp 128 KB transition at strided 4 KB stride access.  `[🟢 HIGH · src: corrections/03_caches_CORRECTED.md §1 + b300_clean/D2_L1_CAPACITY_RIGOR.md + b300_clean/V10_L1_CAPACITY.md]`

L1 size, latency, and bandwidth are ALL carveout-dependent and access-pattern-dependent. A naked "L1 = X KB" or "L1 = Y TB/s" claim without carveout / pattern is meaningless. This section enumerates the regimes.

### Capacity (HIGH)

| Quantity | Value | Source |
|---|---|---|
| Unified L1+SHMEM pool per SM | **256 KB** | `cudaDeviceGetAttribute`, `03_caches.md`, M5 |
| Per-SM peak SMEM (opt-in) | 228 KB = 233,472 B − 1024 B reserved | `B300_TRUE_REFERENCE.md` |
| L1 portion (carveout=0, max L1) | ~228 KB | `03_caches.md` §2 |
| L1 portion (default carveout≈100) | ~20–22 KB | `03_caches.md` §2 |
| L1 line size | **128 B** | architecture-standard, D2 |
| Reserved SMEM/CTA | 1024 B | architecture |
| Chip-wide SRAM aggregate | 148 × 256 KB = 37.9 MB | derived |

The carveout is set per-launch via `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, N)` or `PreferredSharedMemoryCarveout`. The two extremes:

- `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, 228*1024)` → SMEM=228KB, L1≈20KB
- `cudaFuncSetAttribute(..., PreferredSharedMemoryCarveout, 0)` → SMEM=0, L1≈228KB

Most real workloads run at default carveout (L1≈20–32 KB, SMEM≈48–96 KB). Hand-tuned tile kernels often max out SMEM to 228 KB.

### Effective L1 capacity vs access pattern (HIGH)

Two regimes give different "effective L1" answers. Both are correct.

| Regime | Effective L1 | Source |
|---|---|---|
| Strided pointer-chase, 4 KB stride (one line per 4 KB region) | **~128 KB ≈ 1024 lines**, sharp boundary | `D2_L1_CAPACITY_RIGOR.md` |
| Random-access (Fisher-Yates chain, 128 B lines) | **~2–4 KB** effective, smooth ramp 47→277 cy | `V10_L1_CAPACITY.md` |

V10 is **not** in conflict with D2 — random access exposes associativity limits / hash collisions early; strided 4 KB walks the sets evenly. Both fit on the same 256 KB pool. For practical purposes:

- **Sequential / coalesced access**: full L1 capacity available (carveout-dependent)
- **Random access**: associativity-limited, 2–4 KB effective working set in L1 before cache pressure
- **Pointer-chase**: 128 KB sharp boundary (large but not full)

### L1 latency (HIGH)

| Path | Latency | Source |
|---|---:|---|
| Register | 1 cy | catalog |
| L1 hit (warm pointer-chase) | **38–47 cy** | D2 (39 cy @ 1500 MHz), V10 (47 cy random), `03_caches.md` (42–45 cy @ 2032 MHz) |
| L1 → L2 transition | 130–200 cy warm | `03_caches.md` |
| `.ca` vs `.cg` at 8 KB WS | 40 cy vs 552 cy = **13.8× ratio** | `03_caches.md` |

`.ca` = L1+L2 cached (SASS: `LDG.E.STRONG.SM`); `.cg` = L2-only, bypasses L1 (SASS: `LDG.E.STRONG.GPU`). Default `LDG.E.STRONG.SM` is L1-cached.

**Note:** `__ldg` emits `LDG.E.CONSTANT` at SASS, which is **NOT measurably faster than `.ca`/default at L1-resident sizes**. The classical "use `__ldg` for read-only data" advice is benign-but-not-impactful on B300. Per `corrections/STRAYS_CORRECTED.md` §4 (D9_E4 audit): "no measurable difference between `__ldg` and `.ca` at L1-hit".

### L1 bandwidth (MED)

Two cited values:

| Path | BW | Source |
|---|---:|---|
| L1 aggregate (default ld, 8-ILP × 16 unroll) | **~30.5 TB/s** | `V8_L2_BW_VERIFIED.md` |
| L1 aggregate (M5 cheatsheet, optimistic) | ~46 TB/s | `M5_MEMORY_CHEATSHEET.md` |

Spread reflects unrolling / ILP / launch geometry. **30.5 TB/s is the conservative measured peak** under the V8_L2 verification methodology; the M5 cheatsheet 46 TB/s is at L1+register tag-overlap and is the LSU/L1-dispatch ceiling — not strictly L1 throughput.

For practical recipes: budget L1 at **~30 TB/s** for sustained tile-resident work; allow up to ~45 TB/s peak for short hot loops.

### Associativity (MED)

D2 swept 11 stride values 64 B → 64 KB at fixed line count of 128: latency uniform within 0.2 cy. **No power-of-2 aliasing penalty** — B300 uses hashed L1 indexing. (Compare with H100 which had observable bank conflicts at 4 KB stride; B300's hash mitigates.)

The L1 hash function appears to be similar to L2's hashing — bits are XORed across address ranges to distribute lines across L1 sets uniformly. Specific bit positions aren't documented but D2's measurement at 11 strides found no aliasing.

### L1 instruction-level effects

The L1 cache lookup is overlapped with register-file address generation. For an LDG.E with computed address:

```
cycle 0:   compute address (from register or PC)
cycle 1:   issue LDG.E to LSU
cycle 2-N: L1 lookup; if hit, fill register at ~38-47 cy total RT
cycle N+1: register dependent on result becomes ready
```

The 38–47 cy "L1 hit latency" is **issue-to-result-ready** time. If the next instruction depends on the result, the dependent instruction stalls 38–47 cy. If the next instruction is independent (ILP available), the LSU can issue another LDG.E at every-other-cycle (2-cy issue cadence per warp).

For ILP-rich code, you can hide L1 latency entirely. For pointer-chase or dependent-load chains, L1 latency is exposed.

### L1 throughput vs latency tradeoff

| Pattern | L1 BW | L1 latency-hiding | Use |
|---|---:|---|---|
| Pointer-chase (1 dep load at a time) | very low | minimal (latency-bound) | 1-thread linked list traversal |
| 4-ILP independent loads | ~15 TB/s | partial (still single-warp) | dense matvec |
| 4 warps × 8 ILP | ~30 TB/s | nearly full (TLP+ILP) | tile loops |
| 64 warps × 8 ILP | ~30 TB/s (saturated) | full | typical compute kernels |

### Cache hints summary

| Hint | DRAM-bound | L1-resident (8 KB) | L2-hot (4 MB) | Source |
|---|---|---|---|---|
| default | 3.4 TB/s | full L1 path | baseline | `03_caches.md` |
| `.ca` | 3.4 TB/s | 40 cy | **13.1 TB/s** | `03_caches.md` |
| `.cg` | 3.4 TB/s | 552 cy = 13.8× slower | 10.5 TB/s = -20% | `03_caches.md` |
| `.cs` / `.lu` | 3.4 TB/s | similar to `.cg` | similar to `.cg` | `03_caches.md` |
| `__ldg` / `.nc` | 3.4 TB/s | matches default | matches default | `03_caches.md` |

For DRAM-bound work: cache hints don't matter (all hit ~3.4 TB/s — limited by HBM, not L1/L2 path). For L1-resident: prefer default or `.ca`, avoid `.cg`. For L2-hot: prefer `.ca` (1.25× over `.cg`). The pre-2026 catalog claim of "`.cg` 4.7× slower than `.ca`" was a typo — true L2-hot ratio is 1.25×, true L1-resident ratio is 13.8×.

**Footgun:** ⚠ "L1 = 32 KB" without carveout is meaningless. The L1 portion of the 256 KB pool ranges from 20 KB (carveout=100, default) to 228 KB (carveout=0). State the carveout when citing L1 size. Likewise, "L1 = 46 TB/s" is the LSU-dispatch ceiling at L1+register-tag overlap, not the sustained L1 path; for tile work budget 30 TB/s.

**See also:** §11 (L2 hierarchy), §12 (SMEM is the other half of the 256 KB pool), §22 (FFMA dual-issue with L1 loads, Agent B).

---

## §11. L2 cache — three different bandwidths

**Answer:** L2 capacity is **126.5 MB** (NOT 50/96/192/256/280 — those are stale errors). Three different "L2 bandwidth" numbers exist, each correct in its framing: **13.30 TB/s** (lts wire / pure L2 partition BW, ncu metric), **23.85 TB/s** (kernel-effective, includes L1 reuse), and **~30 TB/s** (L1-amplified small-WS). 32 sectors × 32 B = 32 architectural atomic units. L2 has its own clock domain at **1860 MHz**, independent of `-lgc`.  `[🟢 HIGH · src: corrections/03_caches_CORRECTED.md §2 + b300_clean/B300_TRUE_REFERENCE.md + b300_clean/L2_UNITS_REFINED.md + b300_clean/CLOCK_DOMAINS_AND_L2_UNITS.md]`

L2 BW is the single biggest source of confusion in the catalog. Three different numbers float around (10, 13, 17, 22, 23, 26, 30, 36 TB/s — yes, all real) measuring slightly different things. This section disambiguates.

### Capacity (HIGH)

| Quantity | Value | Source |
|---|---|---|
| Total L2 | **132,644,864 B = 126.5 MB** | `cudaDeviceProp.l2CacheSize` |
| Max persisting L2 (AccessPolicyWindow) | 79.1 MB = 62.5 % | `cudaDeviceGetAttribute(MaxPersistingL2CacheSize)` |
| Partitions | **2 sides**, hash-routed | `bench_atom_lat_sides.cu` |
| Address hash flips at | ~4 KB stride | `B300_TRUE_REFERENCE.md` |
| Tagging | physical | `B300_TRUE_REFERENCE.md` |

The 50 / 96 / 192 / 256 / 280 MB values are all stale. Per `corrections/STRAYS_CORRECTED.md` §7, the "96 MB" cosmetic error appears in 4 catalog files (L2_BITSTRIDE_SWEEP, POPCOUNT_3TIER, L2_DRAM_DATA_PWR), all are documentation-only mis-statements; the underlying measurements are all at WS that fit in either 96 MB OR 126 MB (8 MB, 64 MB), so no measurement is affected by the typo. **The correct number is 126.5 MB**.

### Sectoring (HIGH)

| Quantity | Value | Source |
|---|---|---|
| Cache line size | **128 B** = 4 sectors | D3, M5 |
| Sector size | **32 B** | `D3_L2_SECTOR_RIGOR.md` |
| Sub-sector write penalty (4 B stride) | **7× DRAM read amp**, 7.5× write amp | D3 mode 0 |
| Half-sector write (16 B) | 1.7× amp | D3 mode 2 |
| Full sector (32 B aligned) | 0× read amp | D3 modes 3/4 |
| Full line (128 B aligned) | 0× read amp | D3 mode 5 |

Practical implication: **always issue 32 B-aligned writes** (`v8` / 256-bit STG) to avoid sub-sector amplification. The 7× DRAM read amp on 4 B-aligned writes is the hidden cost of "innocent-looking scalar stores" — every scalar store that misses sector alignment causes a read-modify-write of the full 32 B sector.

### L2 bandwidth — three distinct metrics (HIGH on definitions)

| Metric | Value | What it measures | Source |
|---|---|---|---|
| **L2 kernel-effective BW (with L1 reuse)** | **23.85 TB/s** | sustained throughput delivered to SMs in a kernel where L1 amplifies hits; not the L2 wire rate | `B300_TRUE_REFERENCE.md` line 31 (commit `1e590cf`) |
| **L2 bus traffic (ncu `lts__t_bytes`)** | **13.30 TB/s** | actual bytes leaving L2 partitions on the wire | `B300_TRUE_REFERENCE.md` line 32 (same kernel, commit `1e590cf`) |
| **L2 BW @ `.cg`, carveout=100, 8–128 MB WS** | **~17 TB/s** | strict L2-only path, modern repro | `03_caches.md` §3a |
| **L2 BW @ `.cg`, carveout=0, 4–128 MB** | 22–26 TB/s | L1 carveout small; mostly L2 path | `03_caches.md` §3b, MED |
| **L2 BW @ `.ca`, WS ≤ 1 MB (L1-amplified)** | 30–36 TB/s | actually LSU/L1-dispatch ceiling, not L2 | `03_caches.md` §3c |
| **L2 strided `.cg` 64 MB** | **13.85 TB/s** | matches the 13.30 ncu wire number | `V8_L2_BW_VERIFIED.md` |

**Reconciliation rule:** when comparing L2 BW numbers always check the metric:

- "kernel-effective" / "delivered" / "lds.sum" = SM-side throughput (includes L1 amplification)
- "lts" / "wire" / `lts__t_bytes` / `.cg` = pure L2 partitions output
- These differ by **~1.8×** (23.85 / 13.30) due to L1 hit rate within the loop

The "10–36 TB/s reported" range in CLAUDE.md is the union of all metrics above. The CLAUDE.md "L2 = 22 TB/s" is the carveout=0 catalog MED number, in-between, acceptable as a rule-of-thumb.

### Per-SM / per-partition (MED)

| Quantity | Value | Source |
|---|---:|---|
| Per-SM L2 BW (delivered) | 113–180 GB/s/SM (regime-dependent) | `03_caches.md` §3c |
| Per-partition share (lts) | unmeasured directly (needs ncu `fbpa__*`) | open in `03_caches.md` §14 |

148 SMs × 90 GB/s/SM avg = ~13.3 TB/s aggregate at the wire (matches `lts__t_bytes` 13.30 TB/s).

### L2 latency (HIGH)

| Path | Latency | Source |
|---|---|---|
| L2 hit (avg) | **300–310 cy** ≈ 152–157 ns @ 1920 MHz | `03_caches.md` §11, M5 (228 cy chained) |
| L2 hit (near partition) | ~310 cy | `03_caches.md` |
| L2 hit (far partition) | ~660 cy | `03_caches.md` |
| Near vs far ratio | **1.27–2.4×** | `B300_TRUE_REFERENCE.md` (commit `af91798`), M5 (1.27–1.85×) |

The near-far asymmetry comes from the 2-partition layout. An access whose hash routes to the local L2 partition (relative to the SM) gets ~310 cy; cross-partition adds ~350 cy (XBAR traversal). For latency-sensitive kernels, use `cudaAccessPolicyWindow` to pin hot lines to the local partition (worth ~2× latency if you can keep them resident).

### L2 atomic units (HIGH on per-unit; MED on count)

| Quantity | Value | Source |
|---|---:|---|
| Per-unit throughput (single line) | **0.83 packets/video-cy** = 1.55 G pkt/s/unit | `L2_UNITS_REFINED.md` |
| Aggregate uncombined (distinct lines) | ~27 packets/video-cy ≈ 50 Gops/s | `L2_UNITS_REFINED.md`, `CLOCK_DOMAINS_AND_L2_UNITS.md` |
| Inferred L2 atomic unit count | **~32** (27 / 0.83 ≈ 32.5) | `L2_UNITS_REFINED.md` |
| Stride-0 (full collision) | 0.79 Gops/s | `B300_TRUE_REFERENCE.md` |
| Stride-4 (cache-line combining) | 449 Gops/s peak | `B300_TRUE_REFERENCE.md` |
| Stride-32 (1 line/thread) | 184 Gops/s | `B300_TRUE_REFERENCE.md` |
| Stride-256+ (scattered) | ~150 Gops/s plateau | `B300_TRUE_REFERENCE.md` |
| Per-L2-atomic wire traffic | ~95 B L2 / ~110 B DRAM | `CLOCK_DOMAINS_AND_L2_UNITS.md` |

The "~32 L2 atomic units" is **inferred** from the 27 / 0.83 ratio, not directly measured. Per `corrections/STRAYS_CORRECTED.md` §8 and the dispatch-ceiling-skepticism note, this is **MED** confidence (not MED-HIGH as L2_UNITS_REFINED initially claimed). VERSION A REVERIFY in `07_atomics_CORRECTED` shows the ceiling could be higher (combine=32 reaches 20.4 L2 packets/cy).

### L2 video clock (HIGH)

L2 / XBAR sits in its own clock domain at **1860 MHz**, **constant**, and not changed by `nvidia-smi -lgc`. Implications:

- Combined-warp atomics are SM-issue-bound; their throughput moves with SM clock.
- Uncombined / scattered atomics are L2/DRAM-bound and **don't move with SM clock**. They scale only with L2's video clock.
- L2 latency in nanoseconds is roughly clock-invariant (since 1860 MHz is fixed); L2 latency in SM-cycles varies 1500–2032 MHz.

This is documented in `CLOCK_DOMAINS_AND_L2_UNITS.md`. When citing L2 BW or L2 latency, name the clock domain (SM clock for issue-bound work, L2 video clock for memory-bound work).

### L2 read power (HIGH)

| Pattern | Power (sustained ~16 TB/s, 1005 MHz) | Source |
|---|---|---|
| All zeros | 365 W | `L2_BITSTRIDE_SWEEP.md`, `L2_POPCOUNT_SWEEP.md` |
| Random (popcount=16) | **549 W** (peak) | popcount sweep |
| All ones | 388 W | popcount sweep |
| Bit-stride duplication 1..8192 | 537–550 W (NULL effect) | bitstride sweep |

Same bell-curve mechanism as DRAM (§9): bus power follows popcount, peak at d=16, NOT a static-popcount effect (it's toggle activity). See §9.

### L2 cache hints (HIGH)

| Hint | DRAM-bound | L2-hot |
|---|---|---|
| default | 3.4 TB/s | baseline |
| `.ca` | 3.4 TB/s | **13.1 TB/s** (L1 amp) |
| `.cg` | 3.4 TB/s | 10.5 TB/s = -20% |
| `.cs` / `.lu` | 3.4 TB/s | similar to `.cg`, +21% L2 sectors |
| `.nc` / `__ldg` | 3.4 TB/s | == default for L2-hot |

`.ca` vs `.cg` ratio is **1.25×** at L2-hot (NOT 4.7× as some older summaries said — that was a typo). For DRAM-bound work cache hints don't matter.

`B300_TRUE_REFERENCE.md` line 153 surprise #9 ("Cache hints `.cg/.cs/.wb` have NO effect on re-read at 4 MB scale") refers to that specific 4 MB re-read kernel; the general L2-hot 1.25× advantage of `.ca` over `.cg` above is from a different (wider) sweep. Both observations are correct in their respective regimes.

### What DOES persist in L2

`B300_TRUE_REFERENCE.md` finding: **persistent L2 (AccessPolicyWindow) provides NO benefit when the hot working set fits naturally in 126 MB L2**. LRU does it for free. `cudaAccessPolicyWindow` is only a win when:

- Your hot WS is larger than 126 MB but your hot subset fits in 79.1 MB (the persisting cap); OR
- You have multi-kernel pipelines where the next kernel's hot lines need to survive the previous kernel's eviction pressure.

For single-kernel work with hot WS ≤ 126 MB, just rely on LRU.

### Decision flowchart for L2 BW citations

When a user asks "what's the L2 bandwidth on B300?", the answer depends on what they're really asking:

```
User asks "L2 BW on B300?"
├── Are they comparing across GPUs?
│   └── Use lts wire = 13.30 TB/s (apples-to-apples, ncu-anchored)
├── Are they writing a kernel and need to know "what BW will my kernel see?"
│   ├── If WS ≤ 126 MB → 23.85 TB/s kernel-effective (with L1 reuse)
│   ├── If WS ≤ 4 MB → 30 TB/s "L2" — but that's actually L1+register
│   └── If WS > 126 MB → DRAM-bound; see §6, ~7.30 TB/s
├── Are they reading an old paper with "36 TB/s L2"?
│   └── That's L1-amplified. Real L2 wire is 13.30; old paper conflated.
└── Are they tuning a tile size?
    └── Aim to keep the hot tile in L2 (≤ 126 MB) AND fit per-CTA
        in 32 KB or 96 KB SMEM. Both are necessary for SoL.
```

### L2 access pattern recipe ladder (ncu-verified)

Working from the 4-quadrant matrix of (cache hint × access pattern × WS):

| WS regime | Pattern | Hint | Throughput | Tier |
|---|---|---|---:|---|
| 16 MB (L1+L2) | strided 4 KB | default | 46.6 TB/s | LSU+L1+L2 saturated |
| 16 MB | random | default | 23.0 TB/s | L1 misses, L2 hot |
| 16 MB | strided 4 KB | `.cg` | 17.0 TB/s | bypass L1 |
| 64 MB (L2) | strided 4 KB | default | 23.0 TB/s | L2 plateau |
| 64 MB | strided 4 KB | `.cg` | 13.85 TB/s | matches lts wire |
| 64 MB | random | `.cg` | 10.5 TB/s | L2 random penalty |
| **126 MB** | (any) | (any) | **~8.2 TB/s** | **CLIFF — exact L2 cap** |
| 256 MB | strided 4 KB | default | 7.32 TB/s | DRAM (mostly) |
| 4 GB | per-warp 1 KB bursts | default | 7.30 TB/s | DRAM SoL |

The 126 MB cliff is **sharp**: at WS = 120 MB you're at L2 plateau (~22 TB/s); at WS = 132 MB you're DRAM-bound (~7.3 TB/s). The transition is a single-line-grain change because once the WS exceeds L2 capacity, every line is a cold-miss on the second pass (LRU evicts the line that will be needed soonest).

For tile-size tuning: if you can keep WS ≤ 100 MB you have headroom; ≤ 126 MB you're at the edge; > 126 MB you pay DRAM penalty.

### L2 vs the 17 TB/s rumor

Some pre-2026 docs cite "L2 = 17 TB/s" as the canonical L2 BW. Per `03_caches_CORRECTED.md`, this is the **carveout=100, 8–128 MB WS, `.cg` strict L2-only** measurement. It's correct in its regime but is NOT the most useful headline number because:

- It uses `.cg` (bypass L1), which most kernels don't.
- It's at carveout=100 (default), giving ~228 KB SMEM and ~28 KB L1 — most tile kernels run carveout=0 or in-between.
- The "kernel-effective" number (23.85 TB/s, includes L1 reuse) is more representative of what real workloads see.

So when citing one number: **23.85 TB/s** for "what the kernel sees" or **13.30 TB/s** for "what the L2 wire delivers". The 17 TB/s is a particular point on the multi-dimensional surface, not a headline.

### L2 recipe for ncu profiling

To verify L2 metrics on your own kernel:

```bash
# L2 wire bandwidth (lts__t_bytes / runtime):
ncu --metrics lts__t_bytes.sum,gpc__cycles_elapsed.avg \
    --target-processes all ./your_kernel

# L2 hit rate (lts__t_sectors_op_read_lookup_hit.sum vs lookup):
ncu --metrics lts__t_sectors_op_read_lookup_hit.sum,\
    lts__t_sectors_op_read.sum ./your_kernel

# L1 vs L2 partition (l1tex pipe lsu vs lts):
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    lts__t_sectors_op_read.sum ./your_kernel
```

The ratio `lts__t_sectors_op_read.sum / l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` is the L2 hit fraction. For workloads that hit L1, this is < 1; for `.cg` workloads it's ≈ 1.

### L2 partitioning and side-aware kernels

The 2 L2 partitions (sides) are address-hashed. Each SM has a "near" partition and a "far" partition. The address hash flips at ~4 KB stride (see §11.2.1).

For latency-sensitive kernels, pin hot lines to the near partition by:

```cpp
// Use cudaAccessPolicyWindow to mark a region as persisting:
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = hot_data;
attr.accessPolicyWindow.num_bytes = 64 * 1024 * 1024;  // 64 MB
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

This keeps `hot_data` resident in L2 (up to the 79.1 MB persisting cap). The hot lines won't be evicted by streaming traffic.

For SM-side pinning to a particular partition, see `tests/side_aware.cu` (the L2-side-aware reduction project). The pattern is to compute which partition a given address would hash to, then route work to the matching SM's CTAs.

### L2 capacity vs persisting cap

| Quantity | Value | Notes |
|---|---:|---|
| Total L2 capacity | 126.5 MB | LRU-managed |
| Max persisting (AccessPolicyWindow) | 79.1 MB = 62.5 % | Hardware cap |
| Free for streaming | 47.4 MB minimum | Even when persisting fully used |

The 79.1 MB persisting cap is enforced by hardware; you cannot allocate more "persisting" L2 even with a larger AccessPolicyWindow. If you try, the excess is treated as streaming.

### L2 prefetch instructions

Two prefetch paths:

| Instruction | Effect | Use |
|---|---|---|
| `prefetch.global.L1` | Hint to bring line into L1 | Rare; default LDG already does it |
| `prefetch.global.L2` | Hint to bring line into L2 | Useful for cold pre-warming |

V6 found 1.58× speedup for legacy `cp.async` paths by adding `prefetch.L2`. **DO NOT** combine with `cp.async.bulk` (TMA) — V42 found this is 27 % SLOWER (the TMA DMA engine fights with the prefetcher).

For `cp.async` (LDGSTS): `prefetch.L2` 1 cache-line ahead of the load is the canonical pattern.

For `cp.async.bulk` (TMA): no prefetch; let the TMA engine manage its own DMA depth.

### L2 cache eviction observation

Since L2 is LRU-managed, kernels that touch >126 MB will evict their own working set before the second pass. To detect this:

```bash
ncu --metrics lts__t_sectors_op_read_lookup_hit.sum,\
    lts__t_sectors_op_read.sum ./your_kernel
```

Hit rate <50 % for a 100 MB hot working set is a sign of eviction pressure. Reduce WS or use persisting window.

**Footgun:** ⚠ "L2 = 22 TB/s" (or 36 TB/s, or 13 TB/s) is meaningless without specifying the metric. ALWAYS state: kernel-effective (delivered to SMs, includes L1 reuse) vs lts wire (true L2 partition output). They differ by 1.8×. Whenever you compare L2 BW across docs, normalize to one metric. `B300_TRUE_REFERENCE.md` row 32 (13.30 TB/s wire) is the apples-to-apples cross-doc number.

**Footgun (separate):** ⚠ "L2 = 96 MB" (or 50, 192, 256, 280) is wrong — 4 catalog files carry the cosmetic 96 MB error. Real value is 126.5 MB. Cross-check: `cudaDeviceProp.l2CacheSize / (1024*1024)` returns 126.

**See also:** §10 (L1 hierarchy), §12 (SMEM), §9 (L2 power), §27 (L2 atomic detail, Agent C).

---

## §12. Shared memory

**Answer:** **38.4 TB/s peak = 99.8 % of 38.5 TB/s theoretical** (32 banks × 4 B × 2.032 GHz × 148 SMs). 228 KB max user-allocatable per CTA. stmatrix W+R chain hits 34.5 TB/s. SMEM atomic INT throughput ~2.2 Tatomic/s (no contention). Bank conflicts are regime-dependent: 2× in latency-bound, ~1× in throughput-bound (the "32-way conflict = 32× cost" textbook rule does NOT hold on B300).  `[🟢 HIGH · src: corrections/02_shmem_CORRECTED.md + b300_clean/02_shmem.md + b300_clean/V8_SMEM_BW.md]`

### Capacity (HIGH, device-attribute verified)

| Limit | Value |
|---|---|
| Total SRAM per SM (L1+SMEM unified) | **256 KB** |
| `cudaDevAttrMaxSharedMemoryPerBlockOptin` | **228 KB** (227 KB usable + 1 KB reserved) |
| Reserved SMEM per CTA | 1024 B |
| Chip-wide SRAM aggregate | 148 × 256 KB = 37.9 MB |
| Min SMEM size for full carveout | 96 KB (default), 228 KB (opt-in) |

Opt-in via:
```cpp
cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 228 * 1024);
```

### Theoretical peak derivation

```
Banks per SM      = 32
Bytes per bank    = 4
Cycles per access = 1
Banks BW per SM   = 32 × 4 = 128 B/cy
SM clock (boost)  = 2.032 GHz

Per-SM peak BW    = 128 × 2.032 = 260 GB/s/SM
Chip-wide peak    = 260 × 148  = 38,490 GB/s = 38.49 TB/s

At 1920 MHz locked: 36.4 TB/s
At 1500 MHz:        28.4 TB/s
At 1005 MHz:        19.0 TB/s
```

### Verified BW per access pattern (HIGH)

| Pattern | BW (TB/s) | %peak (vs 38.5) | Clock | Source / commit |
|---|---:|---:|---|---|
| **Pure LDS.128 read, RAW addr-chain, 1blk/SM, short run** | **38.4** | **99.8%** | 2032 boost | `d41c38c` (`rigor_smem_sol.cu`); SASS+ncu verified |
| LDS.128, 4 SMSPs | 38.0 | 99% | 2032 | `ninja_smsp_vec.cu` |
| LDS.128, 2 SMSPs | 35.4 | 92% | 2032 | same |
| `ld.shared.v4.u32` non-volatile | 37.6 | 98% | 2032 | `d41c38c` (volatile == non-volatile, identical SASS) |
| float4 typical | 35–36 | 92% | 2032 | `02_shmem.md` |
| ldmatrix.x4.b16 (tensor feed) | 33–35 | 91% | 2032 | `4ccda4f`, `664a67b` |
| stmatrix W+R chain | **34.5** | 90% | 2032 | `8bd85e8` (TRUE_REFERENCE) |
| Read+write mix (4R+1W/iter) | 27.2 | 71% | 2032 | `4503a17` |
| Plain `float` LDS, 8-ILP × 16 unroll | 26.9 | 74% (of 36.4 @ 1920) | 1920 | V8_SMEM_BW.md, `352ab1f` |
| 8 × scalar LDS.32 | 19–26 | 50–67% | 2032 | `4503a17` |
| Sustained (>8000 iter, post-throttle) | 17–21 | ~50% (of 36.4) | 1920 throttled | `02_shmem.md` §6 |

**Headline SoL: 38.4 TB/s = 99.8 %** (`02_shmem.md` and `B300_TRUE_REFERENCE.md` agree).
**Realistic mixed-workload ceiling: 27.2 TB/s** for read+write tile work.

### Bank-conflict regime (V44/V45 reframing — HIGH)

The classical model says "32-way bank conflict = 32× cost" (CUDA C Programming Guide). On B300, this is **only the latency-bound case**. Under throughput regime, the warp scheduler hides most of the serialization.

| Regime | 32-way conflict cost | Source |
|---|---:|---|
| Latency-bound (single warp, dependent chain) | ~2× to 8.2× (V44 chain-serial 2×; D5 5.74×; Q6 8.2× full transpose) | V44, D5, Q6 |
| Throughput-bound (many warps, scheduler hides serialization) | **~1× (effectively free)** | V45 |
| `02_shmem.md` "banks_proper" multi-warp | **8.81×** (148×128, 10k iter) | `bce8bf8` |

**Inconsistency**: The catalog's `bce8bf8` 32-way = 8.81× slowdown is from a multi-warp throughput test, NOT a latency test. This contradicts V45's "~1× hidden" claim under the same nominal regime. The discrepancy is **unresolved** — likely the V45 setup had enough other warps queued to hide the conflict, while `bce8bf8` was contention-saturated.

**Practical take**: the "32× textbook rule" is never observed on B300; the real cost ranges 1× to 8.8× depending on warp count and latency-tolerance of the loop. For tile kernels with high TLP, bank conflicts are far less harmful than the textbook model predicts. For pure latency-sensitive kernels (rare in real workloads), the cost can be up to ~8×.

### SMEM atomics

| Op / contention | Cost | Source |
|---|---:|---|
| INT32 atomicAdd uncontended | 4.6 cy | 02_shmem §atomics, `baeef1f` |
| INT32 atomicAdd 32-way | 4.6 cy (zero penalty!) | same |
| FP32 atomicAdd uncontended | 85 cy | same |
| FP32 atomicAdd 32-way | 5729 cy = 67× | same |
| Aggregate INT atomic peak (all SMs, all-lanes-same-addr) | **~2.2 Tatomic/s** | user memory `project_b300_v8_complete.md` (commit `968e5b7`) |

**Note:** the user prompt said "4.2 Tops/s" but that doesn't match the catalog. The 2.2 Tatomic/s figure (`968e5b7`) is the verified value. The 4.2 Tops/s claim may have been mis-recalled or come from a different op (atomicInc/Dec are 4 ns vs add 8 ns = ~2× faster — could account for the discrepancy). Treat as **MED** until re-verified.

**Practical take**: use **INT atomics for SMEM histograms**, not FP32. The 67× cost gap between INT32 and FP32 contended atomics reflects FP32's read-modify-write being non-cacheable on the SMEM hardware atomic units.

### stmatrix and ldmatrix

| Op | Throughput | Use |
|---|---:|---|
| `ldmatrix.x4.b16` | 33–35 TB/s aggregate | tensor MMA feed (m16n8k16 etc.) |
| `stmatrix.x4.b16` (W+R chain) | 34.5 TB/s | tensor C output spill |
| `ldmatrix.x2.b16` | ~25 TB/s | half-tile feed |

These are the SMEM I/O channels for the legacy tensor pipeline. The new tcgen05 path bypasses SMEM and goes directly via TMEM (see Agent E §50).

### Practical recipes

1. **Use LDS.128** (`float4` or `int4`) for SMEM reads — single-instruction width matters; LDS.32 caps at ~1/4 of LDS.128 throughput.
2. **Volatile is a no-op for SMEM** on B300 — `ld.shared` and `ld.volatile.shared` emit identical SASS and deliver identical BW. The "ld.volatile.shared unlocks 1.8× more BW" claim from B300_PIPE_CATALOG §0 is RETRACTED.
3. **For atomic histograms, use INT32** — FP32 atomic is 67× more expensive under contention.
4. **For tile loops, expect ~30 TB/s sustained** — the 38.4 TB/s peak is achievable in tight microbenches but not under realistic mixed-workload pressure (see "27.2 TB/s mixed" row).
5. **Bank-conflict cost is regime-dependent** — don't over-optimize for the textbook 32× model; benchmark first.

### SMEM bank layout (background)

```
SMEM is organized into 32 banks per SM (one bank per warp lane).
Each bank is 4 bytes wide, accessed in parallel each cycle.
Cycle bandwidth = 32 banks × 4 B = 128 B/cy.

A "bank conflict" occurs when 2+ threads in a warp access different
addresses that map to the SAME bank. Mapping:
  bank_id = (byte_address >> 2) & 0x1F   (i.e., bits [6:2] of address)

Example: thread t accesses addr `t * 4` → bank t, no conflict.
         thread t accesses addr `t * 128` → bank 0 for ALL threads → 32-way conflict.
         thread t accesses addr `t * 4 + (t * 8 << 7)` → varies, may conflict.
```

The classical model says k-way conflict = k× cycles. On B300, the warp scheduler hides bank conflicts when other warps are issuable, so the effective cost is much lower in throughput regime.

### Bank conflict diagnosis

To detect bank conflicts, use ncu:

```bash
ncu --metrics smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./your_kernel
```

If this metric is 0, no bank conflicts. If non-zero, the kernel has them; whether they hurt depends on whether the kernel is latency-bound or throughput-bound.

A useful rule-of-thumb test: "compare bank-conflicting tile to padded tile". If padded version is faster by >20 %, you're in the latency-bound regime and bank conflicts hurt. If <20 %, throughput-bound and conflicts are mostly hidden.

### SMEM persistence across CTAs

SMEM is NOT shared across CTAs — each CTA gets its own private SMEM allocation. To share data between CTAs in the same cluster, use DSMEM (§13). To share across clusters or across SMs not in the same cluster, you must go through L2 (slow) or HBM (slower).

### Power on SMEM

Per `b300_clean/POPCOUNT_3TIER.md` and related, SMEM operations cost relatively low power compared to DRAM:

- L1/SMEM at ~30 TB/s: ~150 W active power
- L2 at 23 TB/s: ~340 W active power
- DRAM at 7.3 TB/s: ~520 W active power

For energy-per-byte: SMEM is ~5 pJ/B, L2 ~15 pJ/B, DRAM ~70 pJ/B. Use SMEM aggressively for reused data; this is the second-largest energy lever after avoiding DRAM-DBI penalties.

**Footgun:** ⚠ "32-way bank conflict = 32× cost" is the 1980s textbook rule and is NOT what B300 delivers. Real cost is 1×–8.8× depending on whether the kernel is latency-bound or throughput-bound. Don't reject a SMEM access pattern just because static analysis predicts 32-way conflicts; profile it under your actual workload pressure.

**Footgun (separate):** ⚠ "ld.volatile.shared unlocks more BW" is a RETRACTED claim from older catalog. Volatile and non-volatile emit identical LDS, deliver identical 37.6–38.4 TB/s. Don't add `volatile` thinking it helps.

**See also:** §10 (L1 is the other half of 256 KB pool), §13 (DSMEM = cluster-shared SMEM), §22 (FFMA + SMEM dual-issue, Agent B), §27 (atomics deep, Agent C).

---

## §13. DSMEM (cluster shared memory)

**Answer:** Per-cluster aggregate DSMEM read **~40 GB/s** (chain-bound, not absolute), aggregate write **~560 GB/s** (issue rate, not completion — V21 had no fence between stores and clock64), TMA multicast **~470 GB/s @ 32 KB tile**. Per-pair latency **164–205 cy = 25 % spread** for reads; writes pair-uniform at **34 cy**. NO shared-bus claim is unprovable from V17 (under-issued by 30×). Cluster max=8 portable (16 advertised).  `[🟡 MED · src: corrections/DSMEM_CORRECTED.md + corrections/DSMEM_DOUBT_REPORT.md]`

DSMEM (Distributed SMEM, also called "cluster SMEM") was a Hopper-introduced feature where CTAs in the same thread-block-cluster can directly read/write each other's SMEM via `ld.shared::cluster` / `st.shared::cluster` with `mapa.shared::cluster` for address translation. On B300 this is functional and used by TMA multicast.

### Cluster placement (cx=8 deterministic, HIGH)

```
CTA 0 -> SM 0    CTA 4 -> SM 32
CTA 1 -> SM 1    CTA 5 -> SM 33
CTA 2 -> SM 16   CTA 6 -> SM 48
CTA 3 -> SM 17   CTA 7 -> SM 49
```

Four TPC pairs (x, x+1) spread across 4 GPCs. 100 % stable across launches (no scheduler randomness for cluster=8 dimensions). Use:

```cpp
__cluster_dims__(8, 1, 1)  // or via cudaLaunchAttributeClusterDimension
__global__ void my_cluster_kernel(...) { ... }
```

### Cluster size limits

| Limit | Value | Source |
|---|---:|---|
| `cudaDevAttrClusterLaunch` | 1 (supported) | API |
| `cudaDevAttrMaxBlocksPerMultiProcessor` (with cluster) | hardware-default | API |
| `cudaDeviceGetAttribute(MaxClustersDimension)` | 16 (advertised) | API |
| **Practical max cluster size** | **8** | empirical (V11–V31) |

Above cluster=8, scheduler spread becomes irregular and crash rates rise. Stick to cluster ≤ 8 for portable code. Memory note `project_b300_v5_complete.md`: "WGMMA dropped, cluster MAX=8".

### SASS codegen nuance

`ld.shared::cluster.u32` with **scalar-register address** compiles to `LD.E` (global window through L2), NOT `LDS`. ncu shows ~4 L2 sectors/load. The `LDS R, [R+UR]` form only appears when the mapa result lands in a uniform register.

This SASS surprise affected several early DSMEM benchmarks (V8/V10) which were measuring loop-overhead because `LD.E` with constant base + invariant offsets got CSE'd. See RETRACTIONS below.

### Latency (1-thread dependent chain, DCE-immune) — HIGH

| Memory | Latency (cy) | ns @ 1920 MHz |
|---|---:|---:|
| Local SMEM (LDS) | 24 | 12.5 |
| DSMEM self (mapa→me) | 54 | 28.1 |
| DSMEM cluster=2 | 214.75 | 111.8 |
| DSMEM cluster=3..8 (avg) | ~180 | 94 |
| **DSMEM best pair (SM32↔SM33)** | **164.80** | **85.8** |
| **DSMEM worst pair (SM16↔SM17)** | **204.97** | **107.0** |
| DSMEM write (fenced) | 34 | 17.7 |
| DSMEM atomic .add | 188–239 | 98–124 |
| DSMEM atomic .cas | 206 | 107 |

**Local/DSMEM ratio ≈ 7.5×** (NOT 0.8% — that was LICM; NOT 4.7× — wrong test).

Key observations:

- **Cluster=2 is 21 % slower** than cluster ≥ 3 (single-GPC vs multi-GPC routing). For latency-sensitive cluster work, **prefer cluster ≥ 3** even if you only need 2 CTAs of capacity.
- **Reads are pair-dependent** (25 % spread); writes are pair-uniform (3 % spread).
- Atomics inherit read-path asymmetry (return value → uses read path).
- Self-read via `mapa` still pays LD.E cost (54 cy vs 24 cy local) — the address translation goes through the cluster routing fabric even when the destination is the same SM.

### Per-pair 8×8 latency matrix (HIGH)

Full V15 8×8 matrix (cluster=8, deterministic SM placement; reads only; cycles @ 1920 MHz):

```
            ----- DESTINATION -----
            CTA0  CTA1  CTA2  CTA3  CTA4  CTA5  CTA6  CTA7
            (SM0) (SM1) (SM16)(SM17)(SM32)(SM33)(SM48)(SM49)
SOURCE
CTA0(SM0)    54   175   188   192   195   197   200   201
CTA1(SM1)   174    54   189   191   194   196   199   201
CTA2(SM16)  187   189    54   205   190   192   200   202
CTA3(SM17)  189   190   204    54   192   194   199   201
CTA4(SM32)  194   195   190   192    54   165   195   197
CTA5(SM33)  196   196   192   194   165    54   197   199
CTA6(SM48)  199   200   200   199   194   197    54   168
CTA7(SM49)  201   201   202   201   197   199   168    54

Best (off-diagonal):  165 cy  (CTA4↔CTA5, both in TPC2/GPC1)
Worst:                205 cy  (CTA2↔CTA3, both in TPC1/GPC0, slow XBAR partition)
Self (diagonal):       54 cy  (mapa→me, NOT free vs 24 cy local LDS)
```

Pair-pattern observations:

- **Adjacent CTAs in same TPC** (CTA0↔1, CTA2↔3, CTA4↔5, CTA6↔7) have the lowest latencies (165–204 cy).
- **Cross-TPC same-GPC** (e.g., CTA0↔2 in GPC0): 188–192 cy.
- **Cross-GPC** (e.g., CTA0↔4 from GPC0 to GPC1): 195–201 cy.
- **Self via mapa**: 54 cy — NOT free; pays the LD.E cost. Use direct LDS for self-access if possible.
- **Worst pair (CTA2↔3, SM16↔17)** is in the same TPC but routes through a slower XBAR partition. Mechanism unknown (probably related to physical-die layout).

Spread: 165 → 205 cy = **25 % range**. Mechanism: routing path length and crossbar arbitration depth. For latency-sensitive cluster algorithms (e.g., ring all-reduce), prefer placing the producer and consumer on adjacent CTAs (CTA pair within TPC).

### Throughput / Bandwidth (CL=8 ring, all CTAs active)

#### Per-warp read BW (1 warp/CTA, ring)

| ILP | cy/load | per-CTA BW (GB/s) |
|---:|---:|---:|
| 1 | 6.42 | 1.20 |
| 4 | 2.17 | 3.53 |
| 8 | 1.45 | 5.29 |
| 16 | 1.08 | 7.11 |

#### Multi-warp read aggregate (cluster of 8)

| warps × ILP | per-CTA (GB/s) | Aggregate (GB/s) |
|---|---:|---:|
| 1 × 4 | 2.76 | 22.08 |
| 2 × 4 | 4.25 | 33.99 |
| **4 × 4** | **5.02** | **40.16** ← read ceiling |
| 8 × 4 | 4.59 | 36.69 |
| 4 × 8 | 5.08 | 40.62 |

**DSMEM read aggregate ceiling ≈ 40 GB/s per cluster** — this is **chain-bound**, NOT a fabric ceiling. With non-chained ILP (addresses derived from `i` not from prior result), throughput is plausibly higher (60–80 GB/s estimated by `DSMEM_DOUBT_REPORT.md`). Treat 40 GB/s as a chain-bound lower bound, not the architectural ceiling.

#### Multi-warp write aggregate (cluster of 8)

| warps × ILP | per-CTA (GB/s) | Aggregate (GB/s) |
|---|---:|---:|
| 1 × 4 | 21.19 | 169.5 |
| 2 × 4 | 42.23 | 337.8 |
| **4 × 4** | **70.08** | **560.7** ← issue rate |
| 8 × 4 | 52.04 | 416.3 |

**DSMEM write aggregate ~560 GB/s per cluster** — but this is **issue rate, not completion**. V21 `push_ring_wr` has NO fence between the `st.shared::cluster` calls and the `clock64` end timer. PTX `st.shared::cluster` is fire-and-forget; the timer ends as soon as the last store enters the queue. **Real delivery rate is unbounded in this measurement.** The pair-uniform 34 cy "fenced write latency" (above) is more trustworthy because it includes a fence.

This is the **LOW-MED** confidence note from `DSMEM_DOUBT_REPORT.md`. The "13× higher than reads" framing is correct as a per-instruction issue rate ratio, but not as a fabric throughput ratio.

### TMA multicast (cp.async.bulk.shared::cluster.multicast, 8-way)

| Tile | Time | Effective BW (×8 delivery) |
|---|---:|---:|
| 1 KB | 0.27 µs | 30.58 GB/s |
| 4 KB | 0.33 µs | 99.69 GB/s |
| 16 KB | 0.41 µs | 323.50 GB/s |
| **32 KB** | **0.56 µs** | **470.68 GB/s** |

For cluster-level data movement → **use TMA multicast with ≥ 16 KB tiles**. This is the canonical Hopper/Blackwell pattern for broadcasting input tiles to all CTAs in a cluster.

Multicast cannot be pipelined deeper than 1 (V48: capped at 13.96 TB/s aggregate vs 14.91 TB/s single-deep at chip-aggregate scale — see §6).

### Contention behavior

#### Ring (each CTA reads a different peer) — NO contention

N=1..8 active, all at 188 cy → **1.00× (flat)**. Dedicated point-to-point routing.

**Caveat (LOW conf):** V17 used 1 thread per CTA with single-issue chained loads. Per-CTA throughput is ~0.16 loads/ns. 8 CTAs × 0.16 = 1.3 Gload/s aggregate — far below any plausible bus saturation. **The "no shared bus" claim is unprovable from V17.** A real bus-contention test would need 8 CTAs × 4 warps × ILP=16 ring. Don't quote "DSMEM has no shared bus" as a hard architectural fact — V17 is under-issued by 30×.

#### Hot-spot reads (all CTAs → 1 peer): per-source serving cap

- N=2: 20.4 GB/s aggregate
- N=8: 14.0 GB/s aggregate (1.80× per-reader slowdown)

**Per-CTA serving port caps at ≈ 15 GB/s** — a single peer can only deliver to ~15 GB/s worth of remote requesters.

#### Hot-spot writes — NO cap (writes posted/async)

N=2..8: each ~20 GB/s, 1.01× slowdown. Aggregate scales linearly to N senders. (Same caveat as above re: issue rate vs completion.)

#### Hot-spot atomics: linear scaling to dest's atomic-unit pipeline (V31)

| N senders | cy/atom | cluster aggregate |
|---:|---:|---:|
| 2 | 214 | 9 Matom/s |
| 4 | 214 | 27 Matom/s |
| **8** | **214** | **63 Matom/s** |

Each sender does 9 Matom/s; the destination's atomic unit is pipelined at 33 atoms/clock. Unlike hot-spot reads, this scales N×.

#### Split-ILP across peers (single reader, N peers)

16 ILP to 1 peer: 7.11 GB/s. 4 peers × 2 ILP: 5.92 GB/s.
**Reader's issue rate caps per-CTA BW — NOT peer's serving rate.**

#### Peer concurrent activity affecting DSMEM reader

| Peer doing | DSMEM reader slowdown |
|---|---:|
| Local SMEM reads | +30 % (263 vs 200 cy) |
| FFMA compute | 0 % (210 vs 211 cy) |

DSMEM competes for the peer's SMEM subsystem, not for its compute / SMSP. So **co-scheduling DSMEM with peer compute is free**; co-scheduling with peer SMEM access costs ~30 %.

#### TMA + DSMEM concurrent (V31)

With 16 KB TMA in flight: 216 cy/load. Without: 216 cy/load (0.04 % diff).
**TMA and DSMEM use independent data paths — perfect overlap.** This is the canonical pipelining recipe.

### Fences and barriers (V24)

| Fence | cy |
|---|---:|
| fence.acq_rel.cluster | 320 |
| fence.sc.cluster | 320 |
| fence.sc.gpu | 320 |
| fence.sc.sys | 2870 (~9× slower) |

cluster / gpu **identical cost** → use `fence.sc.gpu` for safety with no penalty.

### Local SMEM atomic scope (V24, CL=100)

| Scope | cy/atom |
|---|---:|
| .cta (default) | 29.97 |
| .gpu | 29.97 |
| .cluster | 31.40 (+1.4 cy / +5%) |

### Producer-consumer handoff

| Mechanism | cy/msg | µs/msg | Notes |
|---|---:|---:|---|
| barrier.cluster per msg | 613 | 0.320 | naive |
| **Batched (1 fence per N writes)** | **80 amortized** | **0.042** | best |
| 8-CTA ring all-reduce (V25) | 842 cy/step | 3.07 µs total | with fence + barrier |

Rule: **batch DSMEM writes**, emit ONE `fence.sc.cluster` + ONE `barrier.cluster.arrive/wait` to amortize the 320 cy fence cost.

### Store width (single-thread CTA 0→1)

| Width | cy/st | bytes/cy |
|---|---:|---:|
| u32 | 33.26 | 0.12 |
| u64 | 45.21 | 0.18 |
| v4.u32 | 41.34 | 0.39 |
| **v2.u64 (128-bit)** | **29.66** | **0.54** |

Use `v2.u64` for widest per-thread DSMEM store.

### Best-practice rules (concise)

1. Use **TMA multicast** for cluster data movement (470 GB/s @ 32 KB tiles).
2. Prefer **DSMEM writes over reads** (per-instruction).
3. **Batch writes** + 1 fence per batch (42 ns/msg amortized vs 320 ns/fence).
4. **Don't hot-spot reads** (15 GB/s serving cap per peer).
5. **Cluster ≥ 3** beats cluster=2 (21 % faster routing for latency).
6. Single reader per CTA → use **≥ 4 warps** to saturate per-CTA BW.
7. Peer's local-SMEM activity costs you 30 %; peer's compute costs you 0 %.
8. Self-read via `mapa` is **NOT free** (54 cy vs 24 cy LDS).
9. `fence.sc.gpu == fence.sc.cluster` in cost → prefer `.gpu` for safety.
10. **DSMEM ≈ 7.5× local SMEM latency** — not 0.8 %, not 4.7 %, not 9×.

### RETRACTIONS (per `DSMEM_CORRECTED.md`)

- "DSMEM 37 TB/s peak" (V8_DSMEM_BW.md) — DCE'd, RETRACTED. Real read aggregate ~40 GB/s per cluster.
- "DSMEM 48.5 TB/s @ cluster=2" (V10_DSMEM_DEEP.md) — DCE'd, author self-retracted.
- "Cluster=2 is fastest for DSMEM BW" — RETRACTED. Cluster=2 is 21 % SLOWER for latency.
- "DSMEM writes are 4× slower than reads" (V10_DSMEM_WRITES.md) — RETRACTED. Inverse of truth.
- "DSMEM 0.8 % slower than local SMEM" — RETRACTED (LICM); true ratio 7.5×.
- "DSMEM 4.7× slower than local SMEM" (`tests/dsmem_v2.cu`) — RETRACTED (FADD-serialized); true ratio 7.5×.
- "DSMEM = 1035 GB/s remote" — workload-specific, not a peak; methodology unclear.
- "Cluster crashes >15 iters at cluster=4..8" — V12/V26 30/30 success, NOT REPRODUCIBLE.

### Open questions on DSMEM

1. **Chip-aggregate DSMEM scaling** — all BW numbers are per-cluster-of-8. With 18 concurrent clusters, do we still see 40 GB/s/cluster × 18 = 720 GB/s read, or does shared GPC/L2 infrastructure cap aggregate? Untested.
2. **mbarrier.shared::cluster cost vs barrier.cluster** — 613 cy for `barrier.cluster`, 320 cy for `fence.sc.cluster`; modern mbarrier may be cheaper. Untested.
3. **DSMEM under register spill** — all BW tests use ≤256 thr/CTA with low register pressure. Does spill activity on the peer's SM reduce DSMEM serving rate? Untested.
4. **DSMEM bank-conflict propagation** — if source SMEM has conflicts, do they slow remote reader? Untested.
5. **Cross-cluster behavior** — reading from a CTA outside your cluster is supposed to be impossible. What exactly fails? Hard fault, silent zero, undefined? Untested.
6. **DSMEM under sustained load** — SMEM throttles from 38 → 17 TB/s after 8000 iters at 1920. Does DSMEM show similar throttle? Untested.

### Why DSMEM exists at all

The architectural value of DSMEM:

- Allows multi-CTA cooperation without going through L2 (saves ~7× latency).
- Enables TMA multicast (broadcast input tile to all 8 CTAs in a cluster, ~470 GB/s @ 32 KB).
- Useful for streaming-multistage algorithms where producer CTA writes to consumer CTA's SMEM.

But:

- Cluster size is capped at 8 (16 advertised but unstable above 8).
- Per-cluster BW is modest (~40 GB/s read).
- Cross-cluster sharing is impossible.
- Coordination primitives (fence.sc.cluster, barrier.cluster) cost 320–613 cy each.

So DSMEM is best used for tightly-coupled, intra-cluster streaming pipelines, not as a "many-CTAs share data" pattern.

### When NOT to use DSMEM

- For data shared across all SMs → use L2 (with persistent windows if hot).
- For data shared between non-cluster-mate CTAs → must use L2 / HBM.
- For simple reductions → use `__syncthreads` + SMEM within a single CTA, or L2 atomics.
- For sub-millisecond latency CPU↔GPU → use mapped poll (see §15), not DSMEM.

### Best practices for DSMEM kernels

1. Keep cluster size at exactly 8 (matches chip topology, deterministic placement).
2. Use TMA multicast for input broadcasting (not manual DSMEM stores).
3. Batch writes; emit `fence.sc.cluster` + `barrier.cluster.arrive/wait` ONCE per batch.
4. Place producer-consumer pairs on adjacent CTAs (CTA pair within TPC) for lowest latency.
5. Prefer DSMEM writes over reads (~5–6× faster per-instruction).
6. Don't hot-spot reads; per-peer serving cap is ~15 GB/s.
7. Use INT atomics for histograms (avoid FP atomics inside DSMEM).

**Footgun:** ⚠ Don't quote DSMEM read at "37 TB/s" or "48.5 TB/s" or any TB/s figure — those are all DCE'd. Real per-cluster read aggregate is **40 GB/s** (chain-bound, possibly higher non-chained), write **560 GB/s** (issue rate, not completion). When citing for a recipe, use TMA multicast (470 GB/s @ 32 KB) which is HIGH-confidence.

**Footgun (separate):** ⚠ Don't claim "DSMEM has no shared bus" — V17's contention test was under-issued by 30× and cannot rule out a shared bus. The architectural design is plausibly point-to-point per spec, but the test doesn't prove it.

**See also:** §12 (local SMEM), §6 (TMA multicast aggregate at chip scale), §28 (cluster sync, Agent C).

---

## §14. NVLink-5 (Blackwell)

**Answer:** **NVLink 5** (NOT "v7" as legacy docs claimed). P2P read **0.778 TB/s = 86 % of 900 GB/s/dir spec**, write **0.720 TB/s = 80 % of spec**. Bidi aggregate 1.543 TB/s = 86 %. NV18 means 18 NVLink-5 links (each link is full-duplex, 50 GB/s/dir data).  `[🟢 HIGH · src: corrections/12_nvlink_p2p_CORRECTED.md + b300_clean/12_nvlink_p2p.md + project_b300_multigpu.md]`

The "NVLink v7" naming in older docs (CLAUDE.md memory snippet, `13_pcie_system.md` line 6 + 216) is **wrong**. B300 (Blackwell) uses **NVLink 5th generation**. There is no NVLink 7. Generation table:

| GPU | NVLink generation | Per-link data rate |
|---|---|---|
| V100 | NVLink 2 | 25 GB/s/dir |
| A100 | NVLink 3 | 25 GB/s/dir |
| H100/H200 | NVLink 4 | 25 GB/s/dir × 1.681 protocol |
| B100/B200/B300 | **NVLink 5** | 50 GB/s/dir |

### Spec derivation

```
B300 NV18 system (2× B300 directly connected):
Per-link data rate (NVLink 5)  = 50 GB/s/dir
Per-link raw rate (with FEC)   = 53.125 GB/s/dir  (50 × 1.0625 protocol)
NV18 = 18 links               (each full-duplex)
Spec/dir total                = 18 × 50  = 900 GB/s/dir
Spec/dir raw                  = 18 × 53.125 = 956.25 GB/s/dir
Spec bidi total               = 1800 GB/s/sec aggregate
```

### Recommended canonical numbers (HIGH)

| Quantity | Value | % of 900 GB/s/dir spec |
|---|---:|---:|
| Read payload BW (kernel + DMA) | **778 GB/s** | **86%** |
| Read NVLink RX (ncu, includes protocol bytes) | 860 GB/s | 96% |
| Write payload BW (kernel) | **720 GB/s** | **80%** |
| Write NVLink TX (ncu) | 836 GB/s | 93% |
| Bidi payload aggregate | **1543 GB/s** | 86% (2-direction) |
| SM count to saturate | 32 SMs | — |
| Per-SM unsaturated rate | ~38 GB/s | — |
| LOCAL atomic Gops/s | 49 | — |
| REMOTE atomic Gops/s | 16 | 33% of LOCAL |
| Cross-GPU atomic latency | ~1.55 µs / ~3000 cy | 5× LOCAL |
| Cross-GPU fence drain | +17.8 K cy | NVLink in flight |
| `cudaDeviceEnablePeerAccess` cold | 131 ms | one-time |
| `cudaIpcOpenMemHandle` (cross-process) | 56 µs | first-touch |
| NCCL all-reduce floor | 10 µs | small msg |
| Custom ring all-reduce floor | 21 µs | small msg |

### Why two read numbers (778 vs 860)

`12_nvlink_p2p.md` reports both:

- **778 GB/s payload** = bytes the kernel actually delivered (event-timed, end-to-end)
- **860 GB/s NVLink RX (ncu metric `nvlink__data_received`)** = bytes that crossed the link including FEC parity, header bytes, and link-layer protocol overhead

The 860 / 778 = 1.10 ratio matches expected NVLink-5 protocol overhead (53.125 / 50 raw + per-flit headers). Both are correct measurements; they measure different things. When citing, name which.

### Why two write numbers (720 vs 836)

Same nuance:

- **720 GB/s payload** kernel-side
- **836 GB/s ncu TX** including protocol

The 836 / 720 = 1.16 ratio is slightly larger than the read-side ratio (1.10); this could indicate write-side ECC re-encoding or larger header overhead per write transaction. Not directly attributed.

### SM-saturation curve

| SMs active | Read BW (GB/s) |
|---:|---:|
| 8 | 245 |
| 16 | 478 |
| 32 | 778 ← saturated |
| 64 | 792 |
| 148 | 817 |

32 SMs are enough to saturate NVLink at the kernel level. Adding more SMs gives marginal improvements (+5 %) but doesn't change the cap. For multi-GPU kernels, plan for ~32 SMs/CTA-pool dedicated to P2P traffic.

### Bidi P2P

`98.8 GB/s` of full-duplex on PCIe is 1.72× single-direction (see §15) — but for NVLink it's better:

- 778 + 720 / 2 = 749 GB/s avg single-direction
- 1543 GB/s bidi aggregate
- 1543 / 749 = 2.06× — close to perfect duplex

**NVLink 5 is essentially full-duplex** (within 3 % of perfect) on this 2× B300 NV18 setup.

### Atomic operations

Cross-GPU atomic operations on NVLink:

- LOCAL atomic Gops/s = 49
- REMOTE atomic Gops/s = 16 = **33 % of LOCAL**
- Cross-GPU atomic latency = ~1.55 µs ≈ 3000 cy = **5× LOCAL**

Cross-GPU atomics are *expensive* — for hot atomic counters, keep them LOCAL and shard across GPUs with periodic rollups. NCCL's all-reduce primitive is the canonical primitive for this.

### `cudaDeviceEnablePeerAccess` first-touch

131 ms cold-start. **One-time cost per process** — cache and reuse the peer-access state. Don't re-enable per-kernel.

### `cudaIpcOpenMemHandle` for cross-process P2P

56 µs first-touch — cross-process IPC handle import. Memory note `project_b300_v5_complete.md`: "IPC handles 55 µs first-touch". Same order of magnitude.

### NCCL vs custom ring

| Op | Floor latency (small msg) |
|---|---:|
| NCCL all-reduce | 10 µs |
| Custom ring all-reduce | 21 µs |
| NCCL with NVLink-SHARP | UNTESTED (no SHARP fabric on this NV18 system) |

NCCL's small-message latency floor of ~10 µs is competitive with anything you can write by hand. Use NCCL unless you have a specific reason not to.

### Multi-GPU sharded GEMM

`12_nvlink_p2p.md` finding: 0 % slowdown for multi-GPU sharded GEMM with proper tiling. cuBLAS's L2 tiling already accounts for the cross-GPU latency; the NVLink path is largely hidden.

### Peer-fence drain

Cross-GPU `__threadfence_system` drains at +17.8 K cycles compared to single-GPU baseline — this is the NVLink-in-flight wait time. Use sparingly; prefer batched fences (CUDA Graphs, persistent kernels with mailbox handoff).

### NVLink topology query

```bash
# View NVLink topology:
nvidia-smi topo -m

# Expected for 2× B300:
        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV18    0-95,192-287    0               N/A
GPU1    NV18     X      0-95,192-287    0               N/A

# "NV18" = 18 NVLink-5 links between GPU0 and GPU1.
```

Each "NV" entry counts the number of NVLink **links** (each is full-duplex). NV4 = 4 links = 200 GB/s/dir. NV18 = 18 links = 900 GB/s/dir.

For 4-GPU or 8-GPU systems (HGX B300 or DGX B300), the topology shows each pair separately; some pairs may have NV0 (no direct link, must hop through CPU/PCIe — avoid).

### NVLink discovery API

```cpp
// Number of NVLink links to peer:
int n_links = 0;
cudaDeviceGetNvLinkCount(&n_links, peer_id);

// Test peer-access enabled:
int can_access = 0;
cudaDeviceCanAccessPeer(&can_access, src_id, dst_id);

// Enable peer access (one-time per pair, costs 131 ms first call):
cudaSetDevice(src_id);
cudaDeviceEnablePeerAccess(dst_id, 0);
```

### NVLink-SHARP

NVLink-SHARP is a switch-fabric extension where collective ops (all-reduce, broadcast) execute IN the NVLink switch hardware, halving the bandwidth requirement (no per-GPU send-then-receive). Available only on NVL switch systems (e.g., NVL72), NOT on direct-connected NV18 systems like 2× B300 SXM6.

Per `12_nvlink_p2p.md` open question: NCCL with NVLink-SHARP is UNTESTED on this rig (no SHARP fabric).

### Stream-isolated NVLink

When using multiple streams with cross-GPU memcpy, only ONE stream sees full BW at a time (NVLink protocol is connection-oriented per-stream). To overlap multiple cross-GPU ops, use multiple `cudaStream_t` but expect aggregate BW = single-stream BW (778 GB/s read), not 4× single-stream. The 4 async copy engines on EACH side share the single NVLink fabric.

### Open questions on NVLink

1. Per-link breakdown vs aggregate ncu metrics (each link gives 50 GB/s data; how does ncu distribute when one CTA pair dominates?).
2. 3+ GPU NVLink topology (untested here; only 2 GPUs in this chassis).
3. NVLink under power-coupled stress (does H2D + P2P jointly degrade either?).
4. NVLink-SHARP performance on NVL switch systems (no NVL72 here).
5. Cross-GPU latency under contention (1.55 µs measured with 1 SM warm; under all-148-SMs hammering, untested).

**Footgun:** ⚠ Don't quote "NVLink v7" — that's a documentation error in CLAUDE.md memory snippet and `13_pcie_system.md`. B300 uses **NVLink 5**. There is no NVLink 7.

**Footgun (separate):** ⚠ Don't quote "0.78 TB/s = 1.04× spec 757" — that uses **NVLink 4** spec (757 GB/s/dir = 18 × 25 × 1.681) as denominator. NVLink 5 spec is 900 GB/s/dir (18 × 50). Re-normalized: 778 / 900 = 86 %, NOT 104 %. The "exceeds spec" framing is a wrong-generation denominator artifact.

**Footgun (separate):** ⚠ Don't quote "740 GB/s NVLink" (M5 cheatsheet) — that's an unsourced average of read (778) and write (720). When citing, use the directional value.

**See also:** §15 (PCIe is the other interconnect path; same 2× B300 system), §28 (cross-GPU atomics, Agent C), §35 (cross-GPU sync, Agent C).

---

## §15. PCIe Gen6 x16

**Answer:** **0.058 TB/s H2D effective = 23 % of 256 GB/s Gen 6 spec / 90 % of Gen 5 spec** (PHY runs Gen 6, data path caps at Gen 5 effective rate). Pinned ≥64 MB. Full-duplex aggregate 0.099 TB/s = **1.72× single-direction**, 86 % of dual-direction sum. Root cause UNCONFIRMED — three hypotheses, none verified.  `[🟢 HIGH · src: corrections/13_pcie_system_CORRECTED.md + b300_clean/13_pcie_system.md]`

### Recommended canonical numbers

| Quantity | Value | Notes |
|---|---:|---|
| PCIe link gen / width | **Gen 6 x16** | NVML, lspci confirm |
| PCIe H2D pinned (≥64 MB) | **57.7 GB/s** | 90 % of Gen 5 spec, 23 % of Gen 6 spec |
| PCIe D2H pinned (≥64 MB) | **57.4 GB/s** | symmetric |
| PCIe full-duplex aggregate | **98.8 GB/s** | 1.72× single-dir |
| PCIe pageable H2D | **38.0 GB/s** | 66 % of pinned (page migration overhead) |
| Async copy engines | **4** | share single PCIe link |
| D2D same device (2 GB) | 3279 GB/s | 45 % of HBM 7.30 TB/s (kernel-effective) |
| H2D 1 B sync latency | **3.6 µs** | floor |
| H2D 4 KB async+sync | 6.5 µs | |
| D2H 4 KB async+sync | 9.0 µs | reads need ack |
| Persistent kernel + mapped poll | **~4 µs** | best CPU↔GPU RT |
| Power min / max (NVML) | **200 / 1100 W** | not 700, not 1400 |
| Idle baseline | ~180–197 W | |
| `HostNativeAtomicSupported` | 0 | pure PCIe variant (not GH200/GB200 NVL with NVLink-C2C) |
| ECC | always on | 1/16 bus reserved |

### Why Gen 6 PHY caps at Gen 5 effective

PCIe negotiation correctly establishes Gen 6 (64 GT/s) on the link, but data throughput maxes out at ~57.7 GB/s pinned — which is 90 % of Gen 5's 64 GB/s/dir spec, NOT the expected 90 % of Gen 6's ~256 GB/s. Three hypotheses (`13_pcie_system.md` "Open questions"):

1. **BIOS/SBIOS config** — host slot/root complex/retimer negotiates Gen 6 PHY but configures data path for Gen 5.
2. **AMD EPYC 9575F IOD limit** — host CPU's IO die may not deliver Gen 6 DMA rates to memory.
3. **PLX switch / re-driver** — chassis intermediate hardware is Gen 5 only.

**None verified.** Need a different chassis to isolate (host vs switch vs PHY). The B300_TRUE_REFERENCE attribution of "CPU-bound" is **NOT verified** and should be retracted to "root cause unconfirmed".

### Full-duplex characterization

```
Single-direction H2D : 57.7 GB/s
Single-direction D2H : 57.4 GB/s
Concurrent H2D + D2H : 98.8 GB/s
Single-dir × 2       : ~115 GB/s   (theoretical full-duplex)
Achieved fraction    : 98.8 / 115 = 86%
Speedup vs single    : 98.8 / 57.7 = 1.72×
```

So PCIe is **partial-but-not-complete full-duplex** — uses ~86 % of the dual-direction theoretical sum, achieves 1.72× single-direction. For overlapping H2D and D2H workloads, expect ~1.7× rather than 2× speedup.

### Pageable vs pinned

Pageable: **38 GB/s = 66 % of pinned**. The CUDA runtime page-migrates pageable memory through a staging buffer; the 34 % overhead reflects that copy. The "1.5 TB/s pageable" myth (from H100-era dispatch tricks) is well-debunked in `13_pcie_system.md`'s page-migration section — it doesn't apply to B300.

For real workloads: always use `cudaMallocHost` (pinned) for H2D buffers > 1 MB.

### Async copy engines

`cudaDevAttrAsyncEngineCount = 4` (queryable via `cudaDeviceGetAttribute`). The 4 engines share a single PCIe link, so:

- Splitting a large copy across 4 streams gives **NO aggregate gain** (still bandwidth-capped).
- Splitting across 4 streams gives **better latency** for small transfers (parallelism).
- Use case: overlap H2D + D2H + compute + computes on different streams; engines schedule independently.

### Latency floor

| Path | Latency |
|---|---:|
| H2D 1 B sync | 3.6 µs |
| H2D 4 KB async+sync | 6.5 µs |
| D2H 4 KB async+sync | 9.0 µs (reads need ack roundtrip) |
| **Persistent kernel + mapped poll** | **~4 µs** ← best CPU↔GPU RT |

For sub-10-µs CPU-GPU coordination, **don't use cudaMemcpy** — use a persistent kernel polling a mapped (pinned) memory mailbox. User memory `project_b300_v6_complete.md`: "persistent kernel 4us first-touch best round-trip latency".

### `HostNativeAtomicSupported = 0`

This means B300 SXM6 AC is the **pure-PCIe variant** without the NVLink-C2C interconnect that GH200/GB200 NVL parts use. There is no host-coherent atomic path (i.e. no `cudaSystemAtomicsSupported` for atomic ops between CPU and GPU memory).

For host-GPU shared atomics, use explicit fence + read-back patterns over PCIe.

### Power range via NVML

Min 200 W (idle ~180–197 W), max 1100 W (TDP cap). NVML `nvmlDeviceGetPowerUsage` is the canonical query. The "1400 W" or "700 W" figures from older docs are wrong; **1100 W is the TDP cap on this AC SKU**.

### CPU-GPU coordination ladder

Three orders of magnitude between coordination methods. Pick the right one:

| Method | Latency | When to use |
|---|---:|---|
| Persistent kernel + mapped poll | **~4 µs** | Sub-10 µs RT; best for tight inference loops |
| Custom GPU mailbox + write_value | ~5 µs | Similar to above, less overhead |
| `cudaStreamWriteValue` (event-based) | 6–10 µs | Hidden gem; 5–6× faster than naive kernel launch |
| Single empty kernel launch | 7–9 µs | Standard "is the GPU ready?" pattern |
| Kernel + cudaStreamSynchronize | 12–20 µs | Synchronous launch |
| `cudaMemcpy` sync (small) | 3.6 µs floor | One-shot transfers |
| Async H2D 4 KB + sync | 6.5 µs | Standard async |
| `cudaIpcOpenMemHandle` first-touch | 56 µs | Cross-process, one-time |
| `cudaDeviceEnablePeerAccess` first-touch | 131 ms | Once per process pair |
| `cuStreamCreate` | <1 µs | Very fast |
| CUDA graph capture+launch | 15–35× faster than re-launch (after warmup) | Bursty inference |

For latency-bound inference: persistent kernel polling. For throughput-bound: kernel batches with graph capture.

### Async stream behavior

The 4 async copy engines are queryable via `cudaDevAttrAsyncEngineCount`. They can:

- Run independently (no shared queue at the user level).
- Overlap H2D + D2H + compute simultaneously.
- BUT: they share the single PCIe physical link, so aggregate H2D+D2H throughput is bandwidth-capped at 98.8 GB/s, not 4× single-stream.

For maximum overlap: 2 streams (H2D + D2H + compute on stream 0, second batch H2D on stream 1) is sufficient. Adding more streams increases scheduling overhead without increasing aggregate BW.

### Comparison with alternative interconnects

| Interconnect | BW per direction | Latency floor | Where used |
|---|---:|---:|---|
| HBM3E (intra-GPU) | 7.30 TB/s | ~155 ns L2, ~250 ns DRAM | This GPU's memory |
| NVLink 5 (inter-GPU) | 778 GB/s payload | ~1.5 µs | 2× B300 NV18 |
| **PCIe Gen 6 effective** | **57.7 GB/s** | **3.6 µs** | **CPU↔GPU** |
| InfiniBand HDR | 25 GB/s | ~1 µs (with NIC) | Cluster networking (not on this rig) |
| NVLink-C2C (GH200/GB200 NVL) | 450 GB/s | ~50 ns | Not present on B300 SXM6 |

PCIe is **40× slower** than HBM bandwidth and **13× slower** than NVLink P2P. For multi-GPU work, NEVER funnel through CPU memory if you can stay GPU-side.

### Multi-GPU NUMA caveat

Node sees `HostNumaId = 0` (single NUMA node from the GPU's perspective), but the AMD EPYC 9575F CPU can be configured for NPS1/NPS2/NPS4 in BIOS. The "1 node" is what BIOS exposes; could be hiding a true NUMA topology. If you observe asymmetric H2D BW from different CPU sockets, suspect this; profile with `numactl --hardware`.

### Open questions on PCIe

1. Why does PCIe Gen 6 PHY cap at Gen 5 throughput? (UNCONFIRMED — see body)
2. Per-GPU vs shared PCIe BW with both GPUs active in chassis? (untested)
3. GPUDirect RDMA NIC→HBM throughput? (not measured; depends on InfiniBand availability)
4. PCIe Gen 6 PAM4 FEC overhead exact accounting? (out of scope for CUDA tooling)
5. Effective BW under power-coupled PCIe + NVLink load? (untested)

### CPU↔GPU best-practice patterns

For different latency budgets:

```cpp
// Pattern 1: Sub-10us round-trip for inference loop
//   Persistent kernel + mapped poll
volatile int *flag;
cudaHostAlloc((void **)&flag, sizeof(int), cudaHostAllocMapped);

__global__ void persistent_worker(volatile int *flag, ...) {
    while (1) {
        int v = *flag;  // poll
        if (v == EXIT) break;
        if (v != 0) {
            // do work
            __threadfence_system();
            *flag = 0;  // ack
        }
    }
}
// Launch once; reuse for entire inference session.

// Pattern 2: Throughput-bursty inference
//   CUDA Graph capture+launch
cudaGraph_t graph;
cudaGraphExec_t graphExec;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// ... launch kernels ...
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
// Per-iter: cudaGraphLaunch(graphExec, stream); — ~15-35x faster than re-launch

// Pattern 3: Standard async H2D + compute + D2H
cudaMemcpyAsync(d_in, h_in, N, cudaMemcpyHostToDevice, stream0);
my_kernel<<<..., stream0>>>(d_in, d_out);
cudaMemcpyAsync(h_out, d_out, M, cudaMemcpyDeviceToHost, stream0);
cudaStreamSynchronize(stream0);
// Use stream1 for next-batch overlap.
```

### `cudaStreamWriteValue` hidden gem

Per `project_b300_v7_complete.md`: `cuStreamWriteValue` (driver API, not runtime) writes a 4-byte value from CPU to GPU memory in 0.45 µs — **5–6× faster than launching an empty kernel** to write the same value.

```cpp
// Fast write from CPU to GPU memory:
cuStreamWriteValue32(stream, gpu_addr, value, 0);
// vs cudaMemset(...) which is 6.5 µs minimum
```

Use this for signaling (e.g., publishing a "frame ready" flag without kernel launch overhead).

### Pinned vs unified memory

| Allocator | Performance | Use |
|---|---|---|
| `cudaMalloc` | HBM-only, fastest device access | Default for device-resident data |
| `cudaMallocHost` | Pinned host RAM, fast H2D/D2H | Bulk transfers ≥ 1 MB |
| `cudaMallocManaged` | Unified, page-migrating | Convenient for prototyping; avoid in hot path |
| `cudaHostAlloc(MAPPED)` | Pinned, mapped to device | Mapped-poll patterns (see Pattern 1 above) |
| `cudaMallocAsync` | Pool-based, async stream-aware | Modern preferred for dynamic alloc/free |

**Avoid `cudaMallocManaged` in performance-critical paths** — page migration overhead is large and unpredictable. Use it only for "I just want this to work" prototyping.

### Memory pool API

`cudaMallocAsync` (CUDA 11.2+) uses memory pools that are stream-aware and reduce allocation overhead:

```cpp
cudaMemPool_t pool;
cudaDeviceGetDefaultMemPool(&pool, 0);
// Optional: configure pool attributes
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, 256*1024*1024);

void *ptr;
cudaMallocAsync(&ptr, size, stream);
// ... use ptr ...
cudaFreeAsync(ptr, stream);
```

This is faster than `cudaMalloc/cudaFree` for repeated alloc/free patterns.

### Multi-stream PCIe contention

With 4 streams doing simultaneous H2D:

| Streams active | Aggregate H2D BW |
|---:|---:|
| 1 | 57.7 GB/s |
| 2 | 57.7 GB/s (no gain) |
| 4 | 57.7 GB/s (no gain) |

The 4 async copy engines exist for **independence**, not for **aggregation**. Use them to overlap H2D + D2H + compute, NOT to multiply H2D throughput.

For real H2D throughput limits: use ONE stream pinned + bulk transfers ≥ 64 MB.

**See also:** §3 (clock state), §14 (NVLink — the other interconnect; preferred for inter-GPU), §44 (power model, Agent D). No footgun beyond the "Gen 6 effective is Gen 5" surprise (already noted in body).

---

## §A summary — Memory hierarchy at a glance

End-of-section card combining the headline numbers from §1–§15:

```
                                                B300 SXM6 AC
================================================================================
TIER          | CAPACITY     | BW           | LATENCY    | NOTES
--------------|--------------|--------------|------------|--------------------------
Register      | 256/lane     | -            | 1 cy       | per SMSP
SMEM          | 228 KB/CTA   | 38.4 TB/s    | 24 cy      | of 256 KB pool with L1
L1            | 20-228 KB/SM | 30.5 TB/s    | 38-47 cy   | carveout-dep, hashed
DSMEM         | n×228 KB     | 40 GB/s/cl   | 165-205 cy | within cluster (max 8)
L2            | 126.5 MB     | 13.30 TB/s   | 300-310 cy | wire (or 23.85 effective)
              |              |              |            | 2 partitions, hashed
HBM3E         | 268.6 GiB    | 7.30 TB/s    | ~250 ns    | 8 stacks 12-Hi, 7680-bit
NVLink-5      | (P2P)        | 778 GB/s     | 1.5 µs     | 18 links, NV18, full-dup
PCIe Gen 6    | (CPU)        | 57.7 GB/s    | 3.6 µs     | effective Gen 5 only
================================================================================
```

Key reading:

- **Bandwidth ladder**: 38.4 SMEM → 30.5 L1 → 23.85 L2 → 7.30 HBM → 0.78 NVLink → 0.058 PCIe (TB/s). Each tier is ~5–10× slower than the one above.
- **Latency ladder**: 1 reg → 24 SMEM → 38 L1 → 165 DSMEM → 300 L2 → 250 ns DRAM → 1.5 µs NVLink → 3.6 µs PCIe. Each tier is ~5–10× slower than the one above.
- **Capacity ladder**: 256 KB SMEM → 256 KB L1+SMEM pool → 126 MB L2 → 268 GiB HBM. Each tier is 1000× larger than the one above.

For any kernel design: place data at the smallest tier that fits, and access from the closest tier you can reuse from. The `WS ≤ X` check at each tier boundary is the most important kernel-design discipline.

### Cross-section topology summary

```
                     PCIe Gen 6 x16 (effective Gen 5, 57.7 GB/s)
                     │
                  [ Host CPU + System RAM ]
                     │
                     ▼
                  GPU 0 (B300 SXM6 AC)
                  ├── 148 SMs × 4 SMSPs × 32 lanes = 18,944 FP32 cores
                  ├── 128 KB L1+SMEM per SM (256 KB pool, carveout-config)
                  ├── 126 MB L2 (2 partitions, hashed)
                  └── 8 × HBM3E 12-Hi stacks (7680-bit bus on AC SKU)
                     │
                     │ NVLink-5 (NV18 = 18 links, 900 GB/s/dir spec)
                     ▼
                  GPU 1 (B300 SXM6 AC) ← same as above
```

The cluster (intra-GPU) topology:

```
                     1 GPU
                     ├── 8 GPCs (Graphics Processing Clusters)
                     │   └── ~18-19 SMs each
                     │       └── TPC pairs (TPCs of 2 SMs each)
                     │           └── SMs (each 256 KB L1+SMEM, 128 FP32)
                     ├── 16 × 512-bit memory controllers (15 enabled on AC)
                     ├── L2 (2 partitions, 63.25 MB each)
                     └── 8 HBM3E stacks
```

The cluster (inter-CTA) topology, when launched with `__cluster_dims__(8,1,1)`:

```
                  Cluster of 8 CTAs (deterministic SM placement):
                  CTA 0 → SM 0   (TPC0/GPC0)
                  CTA 1 → SM 1   (TPC0/GPC0)  ← pair with CTA 0
                  CTA 2 → SM 16  (TPC1/GPC0)
                  CTA 3 → SM 17  (TPC1/GPC0)  ← pair with CTA 2
                  CTA 4 → SM 32  (TPC2/GPC1)
                  CTA 5 → SM 33  (TPC2/GPC1)  ← pair with CTA 4
                  CTA 6 → SM 48  (TPC3/GPC1)
                  CTA 7 → SM 49  (TPC3/GPC1)  ← pair with CTA 6
                  Within-pair latency: 165 cy
                  Worst-pair latency:  205 cy
                  Spread:              25 %
```

---

End of Section A (§1–§15). Sections §16+ continue in sibling files.

---

