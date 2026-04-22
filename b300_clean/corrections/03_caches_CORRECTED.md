# 03 Caches — CORRECTED Reference (L1, L2, TMEM)

**Scope:** Reconciliation of every L1 / L2 / TMEM capacity, bandwidth, latency,
and sector claim across the b300_clean catalog. Built from a sweep of:
`03_caches.md`, `D2_L1_CAPACITY_RIGOR.md`, `D3_L2_SECTOR_RIGOR.md`,
`V10_L1_CAPACITY.md`, `V8_L2_BW_VERIFIED.md`, `L2_BITSTRIDE_SWEEP.md`,
`L2_POPCOUNT_SWEEP.md`, `L2_UNITS_REFINED.md`, `CLOCK_DOMAINS_AND_L2_UNITS.md`,
`L2_DRAM_DATA_PWR.md`, `B300_TRUE_REFERENCE.md`, `V41_V48_FINDINGS.md`,
`06_tensor_cores.md`, `M5_MEMORY_CHEATSHEET.md`.

System: B300 SXM6 AC (sm_103a, 148 SMs), CUDA 13.2 / driver 580.126.09.
Clock context (matters!): boost ≈ 2032 MHz; `nvidia-smi -lgc 2032` paradox
pins to ~1920 MHz; "video"/L2/XBAR clock is **independent** at 1860 MHz;
HBM3E at 3996 MHz.

Confidence keys: HIGH = SASS+ncu cross-checked, multiple agents agree.
MED = single source, plausible vs theoretical. LOW = unreproduced.

---

## 1. L1 + SHMEM Unified Cache

### 1.1 Capacity (HIGH)

| Quantity | Value | Source |
|---|---|---|
| Unified L1+SHMEM pool per SM | **256 KB** | `cudaDeviceGetAttribute`, `03_caches.md`, `M5_MEMORY_CHEATSHEET.md` line 14 |
| Per-SM peak SHMEM (opt-in) | **228 KB** = 233,472 B − 1024 B reserved | `B300_TRUE_REFERENCE.md` line 137 |
| L1 portion (carveout=0, max L1) | ~228 KB | `03_caches.md` §2 |
| L1 portion (default, carveout≈100) | ~20–22 KB | `03_caches.md` §2 |
| L1 line size | 128 B | architecture-standard, `D2_L1_CAPACITY_RIGOR.md` |

**An L1 size claim without stating carveout is meaningless.** The "L1 = 32 KB"
or "L1 = 128 KB" numbers seen in older docs are all valid at their respective
carveout points. Range across carveout: 20–228 KB.

### 1.2 Effective L1 capacity vs access pattern (HIGH)

Two regimes give different "effective L1" answers; both are correct:

| Regime | Effective L1 | Source |
|---|---|---|
| Strided pointer-chase, 4 KB stride (one line per 4 KB region) | **~128 KB ≈ 1024 lines**, sharp boundary | `D2_L1_CAPACITY_RIGOR.md` |
| Random-access (Fisher-Yates chain, 128 B lines) | **~2–4 KB** effective, smooth ramp 47→277 cy | `V10_L1_CAPACITY.md` |

V10 is **not** in conflict with D2 — random access exposes associativity
limits / hash collisions early; strided 4 KB walks the sets evenly. Both
fit on the same 256 KB pool.

### 1.3 L1 latency (HIGH)

| Path | Latency | Source |
|---|---|---|
| Register | 1 cy | catalog |
| L1 hit (warm pointer-chase) | **38–47 cy** | D2 (39 cy @ 1500 MHz), V10 (47 cy random), `03_caches.md` (42–45 cy @ 2032 MHz) |
| L1 → L2 transition | 130–200 cy warm | `03_caches.md` |
| `.ca` vs `.cg` at 8 KB WS | 40 cy vs 552 cy = **13.8× ratio** | `03_caches.md` (proves L1 path is real) |

`.ca` = L1+L2 cached, `.cg` = L2-only (bypass L1). Default `LDG.E.STRONG.SM`
is L1-cached; only explicit `.cg` (`LDG.E.CG`) bypasses L1.

### 1.4 L1 bandwidth (MED)

| Path | BW | Source |
|---|---|---|
| L1 aggregate (working set fits in L1) | **~30.5 TB/s** | `V8_L2_BW_VERIFIED.md` (default ld) |
| L1 aggregate (per memory cheatsheet) | ~46 TB/s | `M5_MEMORY_CHEATSHEET.md` (older, optimistic) |

Spread reflects unrolling / ILP: 30.5 TB/s is the conservative measured peak.

### 1.5 Associativity (MED)

D2 swept 11 stride values 64 B → 64 KB at fixed line count of 128: latency
uniform within 0.2 cy. **No power-of-2 aliasing penalty** — B300 uses
hashed L1 indexing.

---

## 2. L2 Cache

### 2.1 Capacity (HIGH)

| Quantity | Value | Source |
|---|---|---|
| Total L2 | **132,644,864 B = 126.5 MB** | `cudaDeviceProp.l2CacheSize`, all current docs |
| Max persisting L2 (AccessPolicyWindow) | **79.1 MB = 62.5%** | `cudaDeviceGetAttribute(MaxPersistingL2CacheSize)` |
| Partitions | **2 sides**, hash-routed | `bench_atom_lat_sides.cu`, `03_caches.md` |
| Address hash flips at | **~4 KB** stride | `B300_TRUE_REFERENCE.md` |
| Tagging | physical | `B300_TRUE_REFERENCE.md` |

### 2.2 Sectoring (HIGH)

| Quantity | Value | Source |
|---|---|---|
| Cache line size | **128 B** = 4 sectors | D3, M5 |
| Sector size | **32 B** | `D3_L2_SECTOR_RIGOR.md` |
| Sub-sector write penalty (4 B stride) | **7× DRAM read amp**, 7.5× write amp | D3 mode 0 |
| Half-sector write (16 B) | 1.7× amp | D3 mode 2 |
| Full sector (32 B aligned) | 0× read amp | D3 modes 3/4 |
| Full line (128 B aligned) | 0× read amp | D3 mode 5 |

### 2.3 L2 bandwidth — three distinct metrics (HIGH on definitions)

This is the single biggest source of confusion in the catalog. There are
**three different L2 BW numbers** floating around, each measuring something
different. They are all "right" within their definition.

| Metric | Value | What it measures | Source |
|---|---|---|---|
| **L2 kernel-effective BW (with L1 reuse)** | **23.85 TB/s** | sustained throughput delivered to SMs in a kernel where L1 amplifies hits; not the L2 wire rate | `B300_TRUE_REFERENCE.md` line 31 (commit 1e590cf) |
| **L2 bus traffic (ncu `lts__t_bytes`)** | **13.30 TB/s** | actual bytes leaving the L2 partitions on the wire | `B300_TRUE_REFERENCE.md` line 32 (same kernel, commit 1e590cf) |
| **L2 BW @ `.cg`, carveout=100, 8–128 MB WS** | **~17 TB/s** | strict L2-only path, modern repro | `03_caches.md` §3a (inv 06) |
| **L2 BW @ `.cg`, carveout=0, 4–128 MB** | 22–26 TB/s | L1 carveout small; mostly L2 path | `03_caches.md` §3b, MED |
| **L2 BW @ `.ca`, WS ≤ 1 MB (L1-amplified)** | 30–36 TB/s | actually LSU/L1-dispatch ceiling, not L2 | `03_caches.md` §3c |
| **L2 strided `.cg` 64 MB** | **13.85 TB/s** | matches the 13.30 ncu wire number | `V8_L2_BW_VERIFIED.md` |

**Reconciliation rule:** when comparing L2 BW numbers always check the metric:
- "kernel-effective" / "delivered" = SM-side throughput (includes L1 amplification)
- "lts" / "wire" / `.cg` = pure L2 partitions output
- These differ by **~1.8×** (23.85 / 13.30) due to L1 hit rate within the loop

The "10–36 TB/s reported" range in `CLAUDE.md` is the union of all metrics
above. The CLAUDE.md "L2 = 22 TB/s" is the **carveout=0 catalog MED** number,
which is in-between and acceptable as a rule-of-thumb.

### 2.4 Per-partition / per-SM (MED)

| Quantity | Value | Source |
|---|---|---|
| Per-SM L2 BW | 113–180 GB/s/SM (regime-dependent) | `03_caches.md` §3c |
| Per-partition share | unmeasured directly (needs `fbpa__*`) | open in `03_caches.md` §14 |

### 2.5 L2 latency (HIGH)

| Path | Latency | Source |
|---|---|---|
| L2 hit (avg) | **300–310 cy** ≈ 152–157 ns @ 1920 MHz | `03_caches.md` §11, M5 (228 cy chained) |
| L2 hit (near partition) | ~310 cy | `03_caches.md` |
| L2 hit (far partition) | ~660 cy | `03_caches.md` |
| Near vs far ratio | **1.27–2.4×** | `B300_TRUE_REFERENCE.md` (af91798), M5 (1.27–1.85×) |

### 2.6 L2 atomic units (HIGH)

| Quantity | Value | Source |
|---|---|---|
| Per-unit throughput (single line) | **0.83 packets/video-cy** = 1.55 G pkt/s/unit | `L2_UNITS_REFINED.md` |
| Aggregate uncombined (distinct lines) | ~27 packets/video-cy ≈ 50 Gops/s | `L2_UNITS_REFINED.md`, `CLOCK_DOMAINS_AND_L2_UNITS.md` |
| Inferred L2 atomic unit count | **~32** (27 / 0.83 ≈ 32.5) | `L2_UNITS_REFINED.md`, `B300_TRUE_REFERENCE.md` line 162 |
| Stride-0 (full collision) | 0.79 Gops/s | `B300_TRUE_REFERENCE.md` |
| Stride-4 (cache-line combining) | 449 Gops/s peak | `B300_TRUE_REFERENCE.md` |
| Stride-32 (1 line/thread) | 184 Gops/s | `B300_TRUE_REFERENCE.md` |
| Stride-256+ (scattered) | ~150 Gops/s plateau | `B300_TRUE_REFERENCE.md` |
| Per-L2-atomic wire traffic | ~95 B L2 / ~110 B DRAM | `CLOCK_DOMAINS_AND_L2_UNITS.md` |

### 2.7 L2 video clock (HIGH)

L2/XBAR sits in its own clock domain at **1860 MHz**, **constant**, and
not changed by `nvidia-smi -lgc`. Combined-warp atomics are SM-issue-bound;
uncombined/scattered atomics are L2/DRAM-bound and so don't move with SM clock.

### 2.8 L2 read power (HIGH)

| Pattern | Power (sustained ~16 TB/s, 1005 MHz) | Source |
|---|---|---|
| All zeros | 365 W | `L2_BITSTRIDE_SWEEP.md`, `L2_POPCOUNT_SWEEP.md` |
| Random (popcount=16) | **549 W** (peak) | popcount sweep |
| All ones | 388 W | popcount sweep |
| Bit-stride duplication 1..8192 | 537–550 W (NULL effect) | bitstride sweep |

Bus power follows a **bell curve in popcount**, peak at d=16. Chunk-level
duplication is NOT a power lever (range only 18 W). Inter-dword toggling
dominates intra-dword popcount in the constant-data special case
(`0x12121212` repeated → 365 W not the 506 W naive popcount predicts).

### 2.9 L2 write traffic & DRAM (HIGH)

| Path | Power | Source |
|---|---|---|
| L2-warm sustained (340 W chip) | ~190 W active | `L2_DRAM_DATA_PWR.md` |
| DRAM cold | 1.97× L2-warm active | same |

Both L2-warm and DRAM-cold reads are **content-independent** (<1% variance
across 11 data patterns) when measured correctly with `.cg` 1024 B/warp loads.

### 2.10 Cache hints (HIGH)

| Hint | DRAM-bound | L2-hot |
|---|---|---|
| default | 3.4 TB/s | baseline |
| `.ca` | 3.4 TB/s | **13.1 TB/s** (L1 amp) |
| `.cg` | 3.4 TB/s | 10.5 TB/s = −20% |
| `.cs` / `.lu` | 3.4 TB/s | similar to `.cg`, +21% L2 sectors |
| `.nc` / `__ldg` | 3.4 TB/s | == default for L2-hot |

`.ca` vs `.cg` ratio is **1.25×** (NOT 4.7× as some older summaries
said — that was a typo). For DRAM-bound work cache hints don't matter.

`B300_TRUE_REFERENCE.md` line 153 surprise #9 ("Cache hints `.cg/.cs/.wb`
have NO effect on re-read at 4 MB scale") refers to that specific 4 MB
re-read kernel; the general L2-hot 1.25× advantage of `.ca` over `.cg`
above is from a different (wider) sweep.

---

## 3. TMEM (Tensor Memory, Blackwell-new)

### 3.1 Capacity (HIGH)

| Quantity | Value | Source |
|---|---|---|
| TMEM per CTA | **256 KB** = 512 columns × 128 lanes × 4 B | `06_tensor_cores.md` §76, `M5_MEMORY_CHEATSHEET.md`, `CURIOSITY_LIST_V4.md` D7 |
| Allocator | bump-pointer, max 1 alloc per CTA, sizes pow2 ∈ {32,64,128,256,512} cols | `06_tensor_cores.md` |
| `tcgen05.alloc` latency (uncontended) | 253 cy | `06_tensor_cores.md` |
| `tcgen05.alloc` latency (chip-wide contention) | ~1030 cy | `06_tensor_cores.md` |
| Idle power | 0 W (no static draw) | M5 line 13 (H7 ref) |

### 3.2 TMEM bandwidth (HIGH after retraction)

| Op | BW | Source |
|---|---|---|
| **Read peak** (`tcgen05.ld 32x32b.x16`) | **~57–65 TB/s chip-wide** | `06_tensor_cores.md` §76, `CURIOSITY_LIST_V4.md` D7 (commit eff12d4) |
| Write peak (`tcgen05.st.32x32b.x4`) | **97–131 TB/s chip-wide** | `06_tensor_cores.md` §76 |
| `tcgen05.shift.down` | 51 cy | D7 |

**RETRACTED**: catalog "830 TB/s" and "295 TB/s" TMEM read claims were
DCE-inflated. Use ~60 TB/s. (`06_tensor_cores.md` line 120, `06_tensor_cores.md` §80.)

### 3.3 TMEM access constraints (HIGH)

- `tcgen05.alloc` / `dealloc` / `relinquish` are **`.sync.aligned`** —
  must be called by ALL warp threads. Putting `alloc` behind `if(tid==0)`
  deadlocks the warp.
- C accumulator (TMEM) costs ~5% of total mma power.

---

## RETRACTIONS (numbers to NOT cite)

| Retracted claim | Where it appears | Correct value | Why wrong |
|---|---|---|---|
| L2 = 256 MB | older catalog | **126.5 MB** | unit confusion |
| L2 = 50 MB | older catalog | 126.5 MB | likely single-side measurement |
| L2 = 280 / 192 / 186 MB | older catalog | 126.5 MB | scope confusion |
| L2 BW = 36 TB/s (pure L2) | older catalog | LSU/L1 dispatch ceiling, not L2 | confused L1 amplification with L2 wire |
| L2 BW = 10 TB/s | older catalog | under-occupied launch (148 CTAs × 128 thr) | TLP under-saturation |
| L1 = 32 KB | older catalog | true L1 = 20–228 KB, depends on carveout | carveout not stated |
| L2 70 MB knee | older catalog | no real knee | TLP-hiding artifact |
| `.cg` 4.7× slower than `.ca` | unknown summary | 1.25× | summary typo |
| Texture faster than `__ldg` | Kepler/Maxwell guidance | **REVERSED on B300**: tex is 2–3× SLOWER | architectural change |
| Persistent L2 always helps | older catalog | **NO benefit when hot fits L2** | LRU does it for free |
| 4097 = memory cliff | older catalog | cuBLAS algo-66 selection cliff (10.2× not 30×) | not a cache effect |
| 30× cliff at 4097 | older catalog | actually 10.2× | over-stated ratio |
| **TMEM read = 830 TB/s** | CATALOG line 1265-1293 | ~60 TB/s | DCE-inflated |
| **TMEM read = 295 TB/s** | CATALOG line 1265-1293 | ~60 TB/s | DCE-inflated |
| **V33 TMA SoL = 10.84 TB/s** | initial V33 | **6.72 TB/s** (V33 single-deep), 7.20 TB/s (V46 pipelined) | L2-cache reuse inflated DRAM-bound test → Rule 3 violation (148% of theoretical HBM is impossible). `V41_V48_FINDINGS.md` line 90 explicitly flags this. The 10.84 TB/s is the L2 wire rate (`L2_BITSTRIDE_SWEEP.md` line 81 `lts__t_bytes` = 10.84 TB/s) — the test was actually L2-bound, not HBM. |

---

## SUSPECT TESTS (potentially Rule-3-violating, need re-verification)

The V33 lesson — "test claimed 148% of HBM peak because the working set
re-fit into L2 instead of forcing DRAM" — implies any L2-bandwidth /
HBM-bandwidth test with a working set < 126 MB and any unrolling /
modulo-stride pattern is at risk.

Candidates to re-audit (from this sweep):

1. `03_caches.md` §3a 256 MB WS row: claims `.cg` BW stays ≈17 TB/s even at
   2× L2 cap. Section 3d explains this as UNROLL=16 covering 78 MB / pass
   — fits in L2. **This is acknowledged as an L2 measurement, not a DRAM
   one**, so labelling is correct, but the row caption "(>L2)" is misleading.
2. `V8_L2_BW_VERIFIED.md` 64 MB WS @ 13.85 TB/s: working set fits in 126 MB
   L2 by design (this is a deliberate L2 measurement) — OK.
3. `L2_BITSTRIDE_SWEEP.md` / `L2_POPCOUNT_SWEEP.md` 64 MB @ 16 TB/s sustained:
   fits in L2, deliberate — OK.
4. `M5_MEMORY_CHEATSHEET.md` line 24 "L1/SMEM ~46 TB/s": optimistic vs
   V8's 30.5 TB/s. Likely a different ILP/UNROLL/per-SM-aggregation
   convention; flag as **MED** until re-derived.
5. `M5_MEMORY_CHEATSHEET.md` line 25 "L2 single-SM 23 TB/s": this is
   per-SM aggregate; matches the 23.85 TB/s kernel-effective number from
   TRUE_REFERENCE within rounding — OK.

---

## UNRESOLVED / OPEN

1. **Carveout=0 vs 100 BW gap on current firmware** — modern repro of the
   22→17 TB/s drop at carveout=100 was not redone at carveout=0. The gap
   may have closed (`03_caches.md` §14).
2. **Per-partition L2 BW under contention** — needs ncu `fbpa__*` per-partition
   breakdown (`03_caches.md` §14).
3. **Best access pattern at WS just below 126 MB** — strided patterns show
   partial misses earlier than expected (`03_caches.md` §14).
4. **Isolated CCTL.IVALL cost** — no PTX exposes it standalone; surprise
   #20 says "CCTL.IVALL doesn't exist on B300" (`B300_TRUE_REFERENCE.md`
   line 164) — so isolated cost is moot, but the L1 invalidation path
   from `fence.gl/sys` is still unmeasured in isolation.
5. **L2 sector-aligned dedup anomaly** at p=256 (`L2_BITSTRIDE_SWEEP.md`
   §4): single-sample 17 W dip vs neighbors. Re-run with 3 trials needed.
6. **TMEM read/write parallel & drain interaction** — partially covered in
   `bench_tmem_*.cu` (D7), full matrix not yet built.
7. **TMEM hierarchy** — how columns map to physical TMEM banks
   (`TCGEN05_PATH_NOTES.md` line 109).

---

## CROSS-DOC CONSISTENCY CALL-OUTS

- `M5_MEMORY_CHEATSHEET.md` line 13 lists "TMEM 256 KB / 38 MB" — the 38 MB
  total is **148 SMs × 256 KB / SM**. Fine, but TMEM is per-CTA not per-SM,
  so the chip-wide 38 MB is only realized with 1 active CTA per SM at full
  occupancy. Worth clarifying.
- `M5_MEMORY_CHEATSHEET.md` line 14 "L1/SMEM 38 MB" likewise = 148 × 256 KB.
- `03_caches.md` quotes L1 hit at 42–45 cy @ 2032 MHz; D2 quotes 39 cy @
  1500 MHz; V10 quotes 47 cy random. All consistent within ±20% across
  clock and access pattern.
- `B300_TRUE_REFERENCE.md` is the highest-confidence source; when it
  conflicts with `M5_MEMORY_CHEATSHEET.md`, prefer TRUE_REFERENCE.
