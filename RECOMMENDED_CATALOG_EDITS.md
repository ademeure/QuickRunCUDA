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

## 🔴 CRITICAL FIXES

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
