# Recommended Edits to `B300_PIPE_CATALOG.md`

> **What this is:** specific edits to make to `B300_PIPE_CATALOG.md` based on the audit. Each entry cites the line number, the wrong text, and the corrected text. In patch-ready format for direct application.
>
> **How to use:** Open `B300_PIPE_CATALOG.md`, jump to each line, apply the edit. Or use `sed` / a script.
>
> **Status legend:**
> - 🔴 **CRITICAL** — readers will get genuinely wrong numbers from the current text
> - 🟡 **CORRECTION** — number off by 5-50%, should be updated
> - 🟢 **REFINEMENT** — wording or framing should be improved but the number is approximately right

---

## 🔴 CRITICAL FIXES — NEWLY ADDED 2026-04-23 (RIGOROUS REPLICATION RESULTS)

### EDIT NEW-A: NVFP4 9.9 PF needs clock context

**Line:** L9303 (and similar)

**Wrong/incomplete text:**
> "kind::mxf4nvf4.block_scale.block16 = 9.9 PFLOPS at K=64"

**Correct text (per `justifications/49_nvfp4.md` rigorous replication):**
> "kind::mxf4nvf4.block_scale.block16 = **9.26 PFLOPS at 1942 MHz observed clock** (= 92.6% of NVIDIA's 10 PF spec, = 98.4% of theoretical at observed clock). Catalog's 9.9 PF assumes 2032 MHz boost which was never observed during sustained runs on this rig. cy/MMA = 128.001 (matches catalog's 128.01 exactly). For citation: state both '9.26 PF measured @1942 MHz / theoretical 9.85 PF @2032 MHz boost'."

---

### EDIT NEW-B: NVFP4 `.block32` ptxas rejection — partly FALSIFIED

**Line:** L9259-L9266

**Wrong text:**
> "ptxas V13.2.78 rejects the codegen: 'Illegal modifier .block32 for instruction tcgen05.mma'"
> "All tested syntax variants (.block_scale, .scale_vec::2X, .kind::mxf4, .kind::mxf8f6f4, both .ws and non-.ws forms, raw PTX assembly) produce 'Arguments mismatch' or 'Illegal modifier' from ptxas 13.2"

**Correct text (per `justifications/49_nvfp4_ptxas_errors.txt`):**
> "ptxas V13.2.78 rejects most variants but `kind::mxf4.block_scale.block32` actually **COMPILES** — emits SASS `UTCOMMA` (without `.BLOCK16` suffix). The kernel then crashes at RUNTIME with 'illegal instruction'. So the situation is more nuanced: SOME forms are ptxas-rejected with 'Illegal modifier' messages, but `kind::mxf4.block_scale.block32` is ptxas-accepted but runtime-rejected. For full audit-grade list of accepted vs rejected variants, see preserved file."

---

### EDIT NEW-C: NVFP4 tcgen05.cp shape `128x256b` claim FALSIFIED

**Line:** L9452-L9457

**Wrong text:**
> "| `128x256b` | ✓ | ✗ | Crashes (illegal memory access, descriptor issue) |"

**Correct text (per `justifications/49_nvfp4_cp_shapes.txt`):**
> "| `128x256b` | ✓ | **✓** | Works fine with 8 KB+ smem buffer (catalog's earlier crash was likely smem under-allocation, not a shape limitation) |"

Bonus: `4x256b` also works on this rig — could be added to the working-shapes table.

---

### EDIT NEW-D: NVFP4 K=64 correctness path is broader than catalog claimed

**Line:** L9444+ ("Key breakthrough: tcgen05.cp.cta_group::1.128x128b correctly copies smem→TMEM")

**Catalog implication:** the `tcgen05.cp + TMEM-A` path is required for the K=64 correctness tests.

**Audit finding (per `justifications/49_nvfp4_correctness15.txt`):** the simpler **smem-descriptor A path** (NOT `tcgen05.cp + TMEM-A`) ALSO produces all 15/15 correct outputs. The catalog should note this is one path among several rather than the only working configuration.

---

### EDIT NEW-E: Kernel launch overhead 2.0 vs 5.7 µs reconciled

**Lines:** L7654 (5.7 µs) and L8917 (2.0 µs) — looked inconsistent

**Reconciliation (per `justifications/22m_launch_overhead.md`):**

The two numbers are NOT contradictory; they reflect different timing modes:
- **2.05 µs** = pipelined (2-event around N launches; QuickRunCUDA default `-T` mode)
- **5.20 µs** = per-iter event recording (`--timesPerRun` mode adds ~3 µs overhead per launch)

Catalog L8395 already mentions this; just be more explicit:
> "Empty kernel launch overhead measured TWO ways:
> - **2.05 µs** = bare cudaLaunchKernel pipelined (with single start/stop event around N launches)
> - **5.20 µs** = per-iter event recording (each launch wrapped in its own start+stop event, adds ~3 µs overhead)
> Use the lower number for "what does my workload pay per launch". Use the higher number when comparing to wall-clock benchmarks that record per-iteration events.
>
> Cluster launch (sizes 1/2/4/8) is identical to single-CTA at 2.05 µs flat — no setup overhead."

Catalog kernel-size table (L8385) reproduces EXACTLY (within rounding) on this rig.

---

### EDIT NEW-F: §22h compute-memory overlap — quantitative correction

**Line:** L8309-L8316

**Wrong text:**
> "| Pure memory load (cold cache) | 522 |"
> "Memory + 16 FFMA = 522 (still hidden), Memory + 64 FFMA = 580 (FFMA budget exceeded)"

**Correct text (per `justifications/22h_compute_mem_overlap.md` 11-SASS-file replication):**

The qualitative claim (FFMA fully hidden by cold-DRAM load) is **CORRECT**. But quantitative numbers DIFFER:
- Cold DRAM (LCG walk + l2flush) is **882 cy / 451 ns**, NOT 522
- Catalog's 522 was probably partial-cold (between cold 882 and warm L2-line 335)
- Free FFMA budget is **~225 FFMAs**, NOT ~16 (the 522 cy → ~16 FFMA derivation was based on the wrong 522 baseline)
- Crossover at N≈225-256 FFMAs (above which ptxas register-spills, contributing to apparent extra latency)

Update the table to:
| Pattern | cy/iter |
|---|--:|
| Pure cold DRAM (LCG, l2flush) | 882 |
| Pure warm L2 line (repeat-stride) | 335 |
| Memory + 8 FFMA | 877 (FFMA fully hidden) |
| Memory + 128 FFMA | 877 (still hidden!) |
| Memory + 224 FFMA | 876 (still hidden — bracket of "free" budget) |
| Memory + 256 FFMA | 1217 (crossover; ptxas register-spills here) |

---

### EDIT NEW-J: §22l grid sync overhead 4245 cy is 2× too pessimistic

**Lines:** L7635-L7648

**Wrong text:**
> "Grid sync cost is ~constant at ~4200 cy = 2.2 μs, regardless of grid size. The cost is dominated by atomic acq_rel (1598 cy) + spin loop on phase var."

**Correct (per `justifications/22l_grid_sync_DEEP.md` rigorous DEEP investigation @1800 MHz locked):**

> "Grid sync overhead by implementation, all at full chip (148 SMs, 1800 MHz locked):
>
> | Implementation | cy/sync | µs |
> |---|--:|--:|
> | NVIDIA `cg::grid_group::sync()` | 2234 | 1.29 |
> | catalog's atomicAdd-with-return | 4245 | 2.21 (catalog claim) |
> | atomicAdd no-return (ptxas → REDG) | 1620 | 0.96 |
> | **Ninja: 32-bit `red.relaxed.gpu` + relaxed spin (sense-reversing)** | **1552** | **0.81** ⭐ best |
>
> Catalog's 4245 cy uses atomicAdd WITH return value, which forces ptxas to emit `ATOM.E.ADD.STRONG.GPU` (1122 cy) + MEMBAR.ALL.GPU (575 cy). Discarding the return value drops to REDG (123 cy) and removes MEMBAR.
>
> NVIDIA's cg::sync conservatively emits MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR + ATOM.E.ADD + LD.E.STRONG.GPU + CCTL.IVALL + YIELD + WARPSYNC.ALL + 3× BAR.SYNC for cluster/async safety. The ninja recipe drops these for basic grid-sync semantics. Ninja correctness verified via ping-pong reduction at grid sizes {8,32,64,132,148}.
>
> **Caveat:** ninja recipe assumes data dependencies are bounded by the atomic counter itself. For arbitrary store-before-sync patterns, use `red.release.gpu` + `ld.acquire.gpu` (~1.34 µs, ~cg::sync speed but explicit).
>
> Note: cy NOT exactly clock-invariant — grows 6-7% from 1500→1920 MHz (L2 atomic unit in separate clock domain)."

This is a major architectural finding for grid-sync optimization. The 2.7× speedup over catalog is ninja-meaningful for grid-sync-heavy workloads.

---

### EDIT NEW-I: §28 compiler-reachable uniform ops — minor addition (UPRMT)

**Line:** L2164

**Original text:**
> "Compiler-reachable uniform ops (verified with CUDA 13.2): UIADD3, UIMAD, UMOV, UISETP, ULOP3.LUT. UFFMA/UFADD/UFMUL still not emitted in CUDA 13.2 either."

**Recommendation (per `justifications/28_compiler_gaps.md` direct SASS audit across 20K+ kernels):**

Add **UPRMT** to the compiler-reachable list. Verified 887 instances of `UPRMT` opcode in real kernel SASS via word-boundary grep.

**Corrected text:**
> "Compiler-reachable uniform ops (verified with CUDA 13.2 across 20K+ preserved SASS files): UMOV, UIADD3, UIMAD, UISETP, ULOP3.LUT, **UPRMT**. UFFMA/UFADD/UFMUL still not emitted in CUDA 13.2 either (0 instances confirmed)."

⚠ **Self-correction note:** an earlier draft of this edit claimed several other uniform ops (UFU, USHF, ULEA, UFLO, ULT, UNC) were also compiler-reachable. That was a REGEX ERROR — `grep -hoE "U[A-Z][A-Z0-9]*[A-Z]\b"` matches substrings inside longer opcodes (e.g., `UFU` inside `MUFU.EX2`). Word-boundary verification with `(^|[^A-Z])${op}\b` shows those are 0. Only UPRMT is real.

---

### EDIT NEW-H: §22f L1/L2 stride probe table FABRICATED (collapses 2 experiments)

**Lines:** L8231-L8249

**Wrong text:**
> "Stride sweep (4096 loads after warm-up):
> | Stride | cy/load | Tier (inferred) |
> | 4B/8B/16B/32B | 56 | L1 hit (warps coalesce to 128B requests) |
> | **64B** | **304** | L1 miss → L2 hit (5.4× JUMP!) |
> | 128B-1024B | 316 | L2 hit |
> Sharp break at 64B stride — beyond this, per-thread loads stop benefiting from warp-level coalescing."

**Why wrong (per `justifications/22f_stride_probe.md` rigorous replication):**
The catalog table essentially fabricates a "sharp cliff" by collapsing two different experiments into one row:
- The "56 cy" floor IS real for stride 4 B (single-thread throughput)
- BUT strides 8/16/32 are NOT 56 cy — they monotonically rise to 87/136/160 cy
- The "304 cy plateau" is the catalog's SEPARATE L2 pointer-chase LATENCY entry (L111: "ld.global L2 = 301 cy"), not a continuation of the throughput sweep
- Single-thread throughput plateau at stride > 32 B is actually **158-163 cy**, NOT 304
- True coalescing unit is **32 B sector**, not 128 B line

**Correct text:**
> "Single-thread `ld.global.ca` throughput at varying stride (after warm-up):
> | Stride B | cy/load |
> | 4 | 56.8 |
> | 8 | 87 |
> | 16 | 136 |
> | 32 | 160 |
> | 64+ | 158-163 (plateau) |
>
> The progression is **monotonic**, not a sharp cliff. The catalog's earlier "sharp 64 B break to 304 cy" was an artifact of combining throughput measurements with the separate pointer-chase L2 latency entry (~301 cy at L111). Use 158-163 cy as the single-thread throughput plateau; use 301 cy as the L2 latency for dependent-chain workloads.
>
> Coalescing unit: **32 B sector**, not 128 B line. Stride > 4 B causes per-sector spillage; the warp-level 128 B 'footprint' claim should be re-stated as '4 × 32 B sectors per warp at stride 4 B'."

This is one of the more impactful fabrications in the catalog because the "sharp cliff" claim drives architectural conclusions (e.g. "ALWAYS use stride ≤ 32 B per lane") that are wrong in spirit.

---

### EDIT NEW-G: §30B atom→REDG SASS attribution OVERSTATED

**Line:** L7160 (or wherever atom→REDG mapping is asserted)

**Wrong text (single SASS attribution):**
> "atom.global.add compiles to REDG.E.ADD.STRONG.GPU"

**Correct text (per `justifications/30B_atomics_FOLLOWUP.md`):**

Direct SASS grep across 20K preserved kernel files shows ALL THREE opcodes are emitted depending on context:
- **REDG.E.ADD** when atomic return value is DISCARDED (semantically `red.add`)
- **ATOMG.E.ADD** when return value is USED with default scope
- **ATOM.E.ADD** for some scoped variants (esp. STRONG.GPU with certain address patterns)

Counts across all preserved kernel SASS:
- REDG variants: 1869 occurrences
- ATOMG.E variants: 4976 occurrences
- ATOM.E variants: 318 occurrences

ncu metric implication:
- If your kernel emits REDG → use `lts__t_sectors_op_red`
- If your kernel emits ATOMG.E or ATOM.E → use `lts__t_sectors_op_atom`
- BEST PRACTICE: capture BOTH counters and add them

CONFIRMED: `atom.f16/bf16 atomicAdd` emits `ATOM.E.CAS.STRONG.GPU` loops (CAS-emulation). Packed `f16x2/bf16x2` emits `REDG.E.ADD.F16x2` natively.

---

## 🔴 CRITICAL FIXES — original

### EDIT 1: Catalog says "8 GPCs"; actually 9 + 1 partial

**Sources where this appears:** `b300_clean/B300_CANONICAL_REFERENCE.md` L482 + likely elsewhere via grep `grep -n "8 GPCs\|10 GPCs" B300_PIPE_CATALOG.md`

**Wrong text:**
> "B300 die (Blackwell Ultra, TSMC 4NP):
> ├── 8 GPCs (Graphics Processing Clusters)
> │   └── Each GPC has 9-10 SMs (varies post-yield)"

**Correct text (per L8835 reference card AND DSMEM exhaustive `%smid` measurement):**
> "B300 SXM6 AC die (Blackwell Ultra, TSMC 4NP):
> ├── **10 GPCs** (9 × 16 SMs + 1 × 4 SMs partial = 148 SMs)
> │   └── Each full GPC = 8 TPCs × 2 SMs = 16 SMs
> │   └── 1 partial GPC has 4 SMs (yield-binned AC SKU)"

Note: the comprehensive reference card at L8835 already has the right number. Other locations (e.g. canonical doc L482) need updating.

---

### EDIT 2: DSMEM "essentially free" claim is FALSIFIED

**Lines:** L7029-L7031, L7012, L7836-L7860 (multiple sections)

**Wrong text (L7012):**
> "DSMEM is ~identical latency to local smem — the cluster interconnect on B300 is essentially free."

**Wrong text (L7029-7031):**
> "DSMEM 23 cy remote vs 25 cy local"

**Wrong text (L7842):**
> "DSMEM bandwidth = 99% of local smem. Cluster size doesn't matter (2/4/8 all identical)."

**Wrong text (L7857-7858):**
> "Load (u32) | 25 cy | 23 cy | 0× (free)"
> "Load (v4) | 170 GB/s/SM | 169 GB/s/SM | 0%"

**Correct text (per justifications/13_dsmem.md + 13_dsmem_exhaustive.md):**
> "DSMEM read latency single-chain = **204-223 cy = ~9× slower than local SMEM** (23 cy). SASS reveals `ld.shared::cluster.u32` compiles to **`LD.E`** (global LSU path), not `LDS` — that's the mechanical reason for the penalty.
>
> DSMEM IS hidable with **8-16 outstanding loads per warp** (ILP=32 → 9 cy/load, close to LDS-equivalent effective cost).
>
> Cluster size DOES matter slightly: c=2 (222 cy), c=4 (207), c=8 (207), c=16 (231). Cluster=16 requires `cudaFuncAttributeNonPortableClusterSizeAllowed`.
>
> Write throughput sustained, fenced: 87-117 GB/s/cluster (depends on stride pattern). Single-cluster SoL scales linearly to c=8: 69 → 139 → 278 GB/s/cluster.
>
> Both DSMEM reads AND writes bypass L2 (0.03-0.05% of traffic; cluster-local interconnect separate from L2 fabric)."

Add a footgun callout: "⚠ The 23 cy / 'essentially free' claim is wrong. Use ILP to hide latency."

---

### EDIT 3: FFMA "uniquely uses BOTH fma sub-pipes simultaneously"

**Line:** L218

**Wrong text:**
> "These are the ones that **uniquely use BOTH fma sub-pipes simultaneously** at 2.00 each → **4.00 warp-inst/SM/cy = 128 SASS/SM/cy = 128 scalar FP32 ops/SM/cy**."

**Correct text (per justifications/01_pipe_topology.md):**
> "Scalar FFMA can use EITHER `pipe_fmaheavy` OR `pipe_fmalite` per cycle, with the scheduler load-balancing across H/L sub-pipes. The aggregate dispatch reaches **4.00 warp-inst/SM/cy = 128 SASS/SM/cy = 256 FP32 FLOPS/SM/cy**.
>
> **Important:** FFMA does NOT issue to BOTH sub-pipes per single instruction. ncu in dual mode shows pipe_fmalite=93% AND pipe_fmaheavy=4.5% — proof of alternation, not simultaneous dual-pipe issue. (Packed FFMA2 IS the instruction that uses both sub-pipes for one inst — see L223+.)"

---

### EDIT 4: __syncthreads formula

**Line:** L116

**Wrong text:**
> "| __syncthreads | 12+2W cy | — | adu | — |"

**Correct text (per justifications/24_latency_table.md):**
> "| __syncthreads | **22+2W cy** | — | adu | — |"

(empirical formula on this rig; the +10 cy fixed barrier-instantiation overhead the catalog formula missed.)

Also fix L74: "__syncthreads at BS=512 = 45 cy" → **54 cy**.

---

### EDIT 5: DFMA latency

**Lines:** L103 (cheat-sheet) and L460 (FP64 detailed table) — INCONSISTENT

**Wrong text (L103):**
> "| DFMA (f64) | **92 cy** | 92 cy (**no ILP**) | fp64 | FFMA, ALU free |"

**Correct text (matches L460 = 63.9 cy):**
> "| DFMA (f64) | **63.9 cy** | 63.9 cy (**no ILP**) | fp64 | FFMA, ALU free |"

Confirmed by audit (justifications/24_latency_table.md). L460 was already correct; just remove L103's wrong number.

---

### EDIT 6: FP8 mma.sync emulated

**Line:** L27 (catalog cheat-sheet)

**Wrong text:**
> "| FP8 tensor via mma.sync | **276 TFLOPS** (emulated, ncu-verified) | …"

**Correct text (per justifications/22_tensor_mma_sync.md):**
> "| FP8 tensor via mma.sync | **309 TFLOPS** (emulated via F2FP+HMMA, anti-DCE verified — earlier 276 was 12% LOW) | …"

Note: the warning in the parenthetical about FADD-artifact in earlier 2336/2247 numbers IS REAL and confirmed by audit (naive test reproduces 2163 TFLOPS due to FADD DCE). Keep that warning, just bump the corrected number from 276 to 309.

---

### EDIT 7: fence costs reconciliation

**Lines:** L114-L116 (cheat-sheet), L2885-L2893 (§30.G), L2914-L2922, L3083-L3088, L3625-L3635

**Multiple inconsistent values currently in catalog.** Single-GPU B300 SXM6 AC authoritative ladder (per justifications/30G_fence.md):

| Fence | This rig single-GPU |
|---|--:|
| `__threadfence_block` (cta) | **8 cy / 3.9 ns @ 2032** |
| `__threadfence` (gl) | **267 cy / 131.5 ns**, +~280 cy first-fence-after-write FIXED (NOT linear "+60 cy/write" as L3084 claims) |
| `__threadfence_system` (sys) | **1727 cy / 850 ns single-GPU** (V54's 2806 was a 2-GPU NVLink rig — the 1.62× difference is one extra coherence round-trip) |

**Recommend:** keep ONE table (in §30.G) with these numbers + the multi-GPU caveat for sys. Delete or mark-superseded the inconsistent values at L2885-L2893, L2914-L2922, L3083-L3088, L3625-L3635 (or label each with the methodology context that produced it: single-warp-empty / 1-SM-many-writes / full-chip-busy-load).

Also: retract the "+60 cy/write linear scaling" claim at L3084 — it's a fixed one-time L2-drain (~280 cy), not linear.

---

### EDIT 8: atomic FP16/BF16 atomicAdd "45× slower"

**Line:** L7160 (per skeptical review T4)

**Wrong text:**
> "atom.f16 and atom.bf16 add are ~45× slower than u32 (1527 vs 34 cy), effectively CAS loops"

**Correct text (per justifications/30B_atomics.md):**
> "atom.f16 and atom.bf16 atomicAdd compile to `ATOM.E.CAS.STRONG.GPU` loops (SASS-verified) and are **~6.3× slower than u32** (NOT 45×). Packed `f16x2` and `bf16x2` PTX atomics ARE native (`REDG.E.ADD.F16x2`) and within 12% of u32.
>
> Bonus: `atom.global.add.f32` is **24% FASTER than u32** chip-wide."

The 45× claim was a unit error.

---

### EDIT 9: atomic scope penalty "31.3×"

**Line:** L7140 (per skeptical review T2)

**Wrong text:**
> "Atomic memory ordering: .relaxed add = 51 cy, .acq_rel.gpu add = 1598 cy (31.3× penalty)"

**Correct text (per justifications/30B_atomics.md):**
> "Atomic ordering scope penalty (apples-to-apples): **2.0-2.2× slower** for .acq_rel vs .relaxed (warp-contend 2.03×, chip-wide 2.22×). The catalog's earlier '31.3× penalty' compared chip-throughput to single-thread chain — apples-to-oranges.
>
> Among scopes (.cta / .gpu / .sys), there is **no penalty for L2-hit data** — the 'FREE for scope qualifier' sub-claim is correct."

---

### EDIT 10: atomic per-warp / coalesced ranking

**Line:** L2708 (per skeptical review E6 / K7)

**Wrong text:**
> "[per-warp atomic hotspot 5× slower than single-address chip-wide]"
> "[per-CTA pattern same as single]"
> "[coalesced unique-per-lane = 0.94 atomics/cy/lane chip-wide]"

**Correct text (per justifications/30B_atomics.md):**
> "Atomic contention ranking by throughput:
> | Pattern | Throughput Gops/s |
> |---|--:|
> | 1 hotspot (single addr, all 18944 threads) | 49.1 |
> | N=2 addresses | **1.69 (29× SLOWER — real anomaly)** |
> | per-warp clean (`addr_idx = warpId`) | **53.7 (1.09× FASTER than 1-hotspot)** |
> | per-CTA pattern | **609 (12.4× FASTER than 1-hotspot)** |
> | coalesced unique-per-lane | **221.4 = 0.023 atom/cy/lane** (NOT 0.94 as catalog claimed — 41× off) |
>
> Note: `atom.global.add` compiles to `REDG.E.ADD.STRONG.GPU` (NOT `ATOM.*`); ncu `lts__t_sectors_op_atom` reports 0 — must use `lts__t_sectors_op_red` for true counts."

---

### EDIT 11: TMA chip-wide bandwidth caveat

**Line:** L2374 / L2381 / similar — "chip-wide TMA 21.9 / 29.2 TB/s"

**Wrong wording:**
> "Chip-wide TMA: 29.2 TB/s / 197 GB/s/SM (8 KB × NT=6 × D=3 batched, L2-resident source)"

**Improvement (per justifications/30_tma_sizes.md + 30_tma_vs_ldg_max_tuned.md):**
> "Chip-wide TMA at 4-8 KB tiles: ~21.9 TB/s **with L2-resident source**. With cold DRAM source, chip-wide TMA caps at ~6.4 TB/s (HBM-bound — same ceiling as LDG).
>
> ⚠ **Footgun**: any TMA chip-wide GB/s number above ~7 TB/s is L2-resident, NOT a DRAM peak. Always cite the source-residency."

For comparison: max-tuned LDG.E.128 also reaches 18.25 TB/s in L2-hit regime (catalog's 13.3 TB/s "L2 wire" claim under-counts by 37-54%). Both paths comparable; TMA wins by ~12% in L2-hit regime when both max-tuned, tied at HBM SoL in DRAM-cold.

---

## 🟡 CORRECTIONS

### EDIT 12: TMA "48 cy size-independent issue floor"

**Line:** L58 (cheat-sheet) and L2249

**Current:** "cp.async.bulk issue rate = 48 cy/inst (size-independent floor)"

**Improvement:**
> "cp.async.bulk issue rate floor: **48-50 cy/inst when batched** (N TMAs onto 1 mbarrier, amortized). Pure single-issue (1 TMA, wait, repeat) is **~65 cy** size-independent for 16 B-8 KB.
>
> The 48 vs 65 discrepancy is the amortization benefit, not size-dependence."

### EDIT 13: TMA "8 KiB sharp crossover"

**Line:** L59

**Current claim is mostly right** but should clarify:
> "The 8 KiB crossover is sharp in the user-facing GB/s/SM metric (jumps 20→40→79→150→241), not in the cy/TMA metric (which is gradual 48.1→48.5→49.6→52.2→65.3 over the 16B-8KB range)."

### EDIT 14: LDS / L1 hit catalog latencies are 14% high

**Line:** L109-L110

**Wrong:** "ld.shared 24 cy" / "ld.global L1 39 cy"
**Catalog L24 says LDS=33 / L1=43.

**Correct (per justifications/24_latency_table.md):**
- LDS hit single-chain = **29 cy** (catalog's 33 is 14% high)
- L1 hit single-chain = **38 cy** (catalog's 43 is 14% high)

### EDIT 15: mbarrier RTT "54 cy"

**Line:** L73 / §24 entry

**Wrong:** "mbarrier RTT (single thread, count=1) | 54"

**Correct (per justifications/24_latency_table.md):**
> "mbarrier RTT (single thread, count=1) = **123 cy** for full arrive+test_wait round-trip. The 54 cy was just the arrive-only dispatch portion."

### EDIT 16: redux.sync.add/or/and/xor latency

**No current catalog entry for the add/or/and/xor variants — just min/max.**

**Add to §24 latency table:**
> "redux.sync.min/max (CREDUX SASS) | 18 cy"
> "**redux.sync.add/or/and/xor (REDUX SASS) | 44 cy (2.4× slower than min/max — different SASS opcode)**"

(SASS-verified: `CREDUX.MIN/MAX` is the compact form, `REDUX.SUM/OR/AND/XOR` is the slower form.)

---

## 🟢 REFINEMENTS

### EDIT 17: Clock state convention

The catalog mixes 1800 / 1920 / 2032 MHz across runs (admits this at L14). On THIS rig, the **DVFS settling clock under sustained load is 1942 MHz** (neither catalog's "1.92 GHz" nor spec's "2032 MHz boost").

**Recommend:** add a note at the top of the catalog:

> "Clock state convention used in this catalog: most cycle counts are clock-independent. TFLOPS / TB/s numbers are tagged with their clock context (`@1920` / `@2032` / `@1500` / `@1005`). On this rig, the DVFS settling point under sustained-FFMA load is 1942 MHz (not 1920 nor 2032). Re-derive % SoL using your own measured clock when in doubt."

### EDIT 18: ncu metric warnings

Add to §31 (methodological notes) or new methodology section:

> "**ncu metric footguns observed in this catalog and corrections:**
> - `sm__sass_data_bytes_mem_shared_op_ld.sum` is **warp-aggregated** (warp_inst × 512 B for LDS.128), NOT per-lane. Naive 16 B/inst accounting undercounts SMEM by 32×.
> - `lts__t_bytes` UNDERCOUNTS LDG L2-hit by 2.7× (MSHR/crossbar dedup). For LDG use `l1tex__t_bytes`; for TMA use `lts__t_bytes`.
> - `lts__t_sectors_op_atom` reports 0 for `atom.add.u32` (which compiles to REDG, not ATOM). Use `lts__t_sectors_op_red` instead. Only CAS variants generate true ATOM sectors.
> - `pipe_tensor` does NOT measure tcgen05.mma — only legacy mma.sync HMMA family.
> - default `LDG.E` hits L1 even for "DRAM" tests unless using `.cg` + Sattolo-shuffled chain over WS > L2 (otherwise L1/L2/DRAM all collapse to ~38 cy)."

### EDIT 19: "FREE" / "no penalty" framing

Catalog uses "FREE", "essentially free", "no penalty", "zero-cost" multiple times. Each is a strong claim; recommend qualifying them:

| Catalog phrase | Where | Refinement |
|---|---|---|
| "DSMEM essentially free" (L7012) | §30.H | DELETE — falsified (see EDIT 2) |
| "scope qualifier FREE for global atomics" (L7131) | §15 | Refine: "FREE for L2-hit data; not tested for DRAM-bound or DSMEM scope" |
| "Predicated execution FREE" (§§ around L8335) | "Predicated Execution Cost (FREE)" header | Quantify with actual cycles (no measurement currently shown) |
| "DSMEM atomics 0× free" (L7857) | §30.H | DELETE (DSMEM atomics are 51 cy = 2.1× slower, also covered in §30.H itself) |

### EDIT 20: Comprehensive Reference Card consistency

Catalog has the "Comprehensive Reference Card" at L8807 which is internally MORE accurate than many earlier sections (correctly says "10 GPCs (9×16+1×4)", correctly says HBM bus 7680 bits, etc.). Recommend hoisting it to the TOP of the catalog as the canonical spec sheet, with the rest of the catalog as deep-dive material that references back to it.

---

## How to apply these edits

Option 1 (manual): `vim B300_PIPE_CATALOG.md`, jump to each line, apply edit.

Option 2 (script-assisted): see `STATUS_OF_REPLICATION.md` for which numbers ARE verified (so you don't need to re-test). Apply EDIT 1-11 (CRITICAL) first; EDIT 12-20 are nice-to-have.

Option 3 (delegate): the changes are small and well-scoped. A new sub-agent could apply all 20 edits in one pass, using this document as the spec.
