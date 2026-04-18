# B300 Cache Hierarchy — Clean Reference (L1 + L2 + Texture/Surface)

**Scope:** L1 + SHMEM unified pool, L2 (size + BW + access patterns), persistent L2 (AccessPolicyWindow), texture/surface paths, cache-hint variants (`.ca`/`.cg`/`.cs`/`.lu`/`.nc`).

**GPU:** NVIDIA B300 SXM6 AC (sm_103a, 148 SMs).
**Clock:** 2032 MHz boost / 1920 MHz under `nvidia-smi -lgc 2032` (use boost for L1/SHMEM tests, lock for memory BW reproducibility).
**Driver:** 580.126.09, CUDA 13.2.

Confidence markers: **HIGH** = SASS+ncu cross-checked, multiple agents agree. **MED** = single agent, plausible vs theoretical. **LOW** = framework-dependent or unreproduced.

---

## 1. Cache topology (HIGH)

| Resource | Size | Source |
|---|---|---|
| L1 + SHMEM unified pool per SM | **256 KB** | `cudaDeviceGetAttribute` + measured carveout sweep (inv 12) |
| L2 cache total | **132,644,864 B = 126.5 MB** | `cudaDeviceGetAttribute(L2CacheSize)` |
| Max persisting L2 (AccessPolicyWindow) | **79.1 MB (62.5% of L2)** | `cudaDeviceGetAttribute(MaxPersistingL2CacheSize)` |
| L2 partitions | **2 sides** (hash-routed) | `bench_atom_lat_sides.cu`; address hash flips ~every 4 KB |
| Per-SM peak SHMEM (opt-in) | **228 KB** (233,472 - 1024 reserved) | catalog |

**Why NOT 256 MB or 50 MB**: `cudaDeviceProp.l2CacheSize` returns exactly 132,644,864 bytes. Earlier "280/192/186 MB" claims in the catalog were unit/scope confusions; "50 MB" was likely a single-side measurement.

---

## 2. L1 + SHMEM carveout (HIGH, inv 12)

`cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout, N)` controls the split.

| Carveout (`-o`) | Measured L1 | Measured SHMEM | Notes |
|---:|---:|---:|---|
| 0 (max L1) | **~228 KB** | ~28 KB | SHMEM hits floor (CUDA RT needs ~28 KB) |
| 25 | **~192 KB** | ~64 KB | Linear |
| 50 | **~128 KB** | ~128 KB | Balanced |
| 75 | **~52-56 KB** | ~200 KB | Carveout snaps in 8 KB steps |
| 100 (max SHMEM) | **~20-22 KB** | ~234 KB | L1 hits floor |

- **L1 hit latency: 42-45 cy at 2032 MHz** (warm pointer-chase). Confirmed by `.ca` (40 cy) vs `.cg` (552 cy) at 8 KB WS — 13× ratio proves L1 path is real.
- **L1→L2 transition latency: ~130-200 cy** (warm L2).
- **DRAM pointer-chase: ~900-1700 cy.**
- **Default carveout favors SHMEM** — uncached `.ca` loads start showing L2 latency at WS as small as 4-8 KB unless you explicitly set `co=0`.

**Resolves catalog "L1 = 32/128/192 KB" claims** — all correct at their respective carveouts. Without stating carveout, an L1 size claim is meaningless.

---

## 3. L2 read bandwidth (HIGH for the *consensus* range; sources differ on exact peak)

L2 BW depends on (working set, occupancy, cache hint, carveout). The catalog has historic claims spanning **10-36 TB/s**; the reconciled answer is:

### 3a. Modern measurement (inv 06, 1920 MHz under `-lgc 2032`)

296 CTAs × 1024 threads (full occupancy), `ld.global.cg.v4.u32` (SASS-verified `LDG.E.128.STRONG.GPU`), UNROLL=16, modulo-stride.

| WS | `.cg` BW | `.ca` BW | L2 hit rate (ncu) |
|---|---|---|---|
| 1 MB | 16.2 TB/s | **20.6 TB/s** | 99.7% |
| 4 MB | 16.8 | 19.4 | 100% |
| 16 MB | 16.8 | 18.6 | 100% |
| 32 MB | 16.2 | 17.8 | 100% |
| 64 MB | 16.6 | 17.8 | 94% |
| 128 MB | 17.4 | 17.6 | 84.5% |
| 256 MB (>L2) | 17.4 | 17.4 | 81.2% |

ncu cross-check at 32 MB: `lts__t_sectors_op_read.sum × 32B = 199.8 TB`, expected 200 TB (99.9% match). ncu-derived L2 BW = 18.1 TB/s; event-based = 17.1-17.4 TB/s. Difference is ncu overhead.

### 3b. Higher numbers from earlier catalog (carveout=0, near-boost clock)

At carveout=0 the larger L1 acts as a BW amplifier: catalog reports **22-26 TB/s plateau** at 4-128 MB and **30.3 TB/s @ 1 MB WS** with `.cg`. These were taken before a likely firmware/power-management change; modern repro converges on 17-18 TB/s at carveout=100 and 22+ TB/s at carveout=0.

### 3c. Reconciled answer

| Condition | L2 BW |
|---|---|
| `.cg`, carveout=100, full occupancy, WS 8-128 MB | **17 TB/s** (modern HIGH) |
| `.cg`, carveout=0, full occupancy, WS 4-128 MB | **22-26 TB/s** (catalog MED) |
| `.ca`, WS ≤ 1 MB (L1-amplified) | **30-36 TB/s** (MED — really LSU/L1-dispatch ceiling, not L2) |
| Per-SM | 113-180 GB/s/SM (depending on regime) |

**Lower bound is reliable; the 30+ TB/s "L1+L2 hybrid" peaks are real but architecture-specific edge cases.**

### 3d. The "L2 plateau is flat at WS > L2" surprise

Even at WS = 256 MB (2× L2 cap), `.cg` BW stays ≈ 17 TB/s. Reason: UNROLL=16 inner loop covers 16 × nthr × 16B ≈ 78 MB of *unique* address per pass, fits in L2; modulo-stride access pattern has more temporal locality than face-value WS suggests. **L2 stays the bottleneck even when "data is in DRAM"**, because HBM3E (7.2 TB/s) refills L2 in parallel with the L2→SM 17 TB/s pipe.

---

## 4. L2 access patterns (MED, l2_access_patterns.cu)

| WS | seq | random | strided-4K |
|---|---|---|---|
| 1 MB (L1) | very high | high | mid |
| 16 MB (L2) | ~17 TB/s | dropoff | gradual |
| 64 MB (L2) | flat ~17 TB/s | dropoff | gradual |
| 126 MB (L2 edge) | flat | falling | falling |
| 256 MB (DRAM) | DRAM-bound | DRAM | DRAM |
| 1 GB (DRAM) | ~7 TB/s | <1 TB/s | <1 TB/s |

**No sharp 60 MB knee** despite 2-side architecture — the L2 address hash mixes both sides at fine granularity (~4 KB blocks) so even WS=8 MB sees ~50/50 partition split. The 70 MB knee in earlier sections was a TLP-hiding artifact, not a capacity wall.

---

## 5. Persistent L2 — AccessPolicyWindow (MED)

`persistent_l2.cu` configures `cudaStreamAttributeAccessPolicyWindow` with `cudaAccessPropertyPersisting` over an 8 MB hot region, then alternates with cold reads.

**Finding (consistent with theory):** Persistent L2 has **zero measurable benefit when the hot data already fits in regular L2**. LRU keeps it resident anyway. Persistent L2 only matters when you have a moderately-sized (~80 MB) hot working set being thrashed by larger cold streams. For most kernels, the access policy window is overhead with no payoff.

---

## 6. Cache-hint loads (MED, `cache_hint_loads.cu`)

| Hint | DRAM-bound (1 GB WS) | L2-hot (4 MB WS) |
|---|---|---|
| `ld.global` (default) | 3.4 TB/s | baseline |
| `.ca` (L1+L2) | 3.4 | **13.1 TB/s** baseline |
| `.cg` (L2 only, skip L1) | 3.4 | **10.5 TB/s = -20%** |
| `.cs` (streaming, evict-on-fill) | 3.4 | similar to .cg, +21% L2 sectors |
| `.lu` (last-use) | 3.4 | similar to .cg |
| `.nc` / `__ldg` | 3.4 | identical to default for L2-hot |
| `.L1::evict_first/last/no_allocate` | 3.4 | within 0.1% |

**Headlines:**
- For DRAM-bound work, **cache hints don't matter** (all within 0.1%).
- For L2-hot work, **`.cg` is 20% slower than `.ca`** because it bypasses L1 amplification (was wrongly summarized as "4.7×" elsewhere — the correct number is ~1.25×).
- `.cs`/`.lu` add 21% extra L2 sector traffic vs `.ca` (visible in ncu).
- `.cs`/`.lu` are pure hints, not bandwidth-improving.

---

## 7. CCTL / prefetch (LOW for cost, HIGH for SASS mapping)

| PTX | SASS | Issue cost | Effect |
|---|---|---|---|
| `prefetch.global.L1 [p]` | `CCTL.E.PF1` | 2 cy dispatch | async; very slow if synced (~2 ms for 128 issues — serializes against memory system) |
| `prefetch.global.L2 [p]` | `CCTL.E.PF2` | 2 cy | async; same concern |
| `applypriority.global.L2::evict_normal [p], 128` | `CCTL.E.DML2` | 2 cy | demote-to-L2 hint |
| `discard.global.L2 [p], 128` | `CCTL.E.RML2` | 2 cy | evict hint |
| `cp.async.bulk.prefetch.L2 [p], 128` | `UBLKPF.L2` | 5 cy | bulk L2 prefetch via uniform pipe |
| `ld.global.L1::evict_last/first` | `LDG.E.EL` / `LDG.E.EF` | 2 cy | normal load + LRU hint |
| `CCTL.IVALL` (from `fence.gl/sys`) | — | unknown isolated cost | invalidates ALL L1 lines |

**`CCTL.IVALL` is the reason `fence.gl`/`fence.sys` are 20-200× more expensive than `fence.cta`** — see fences report. Isolated CCTL.IVALL cost is unverified.

---

## 8. Texture vs `__ldg` (HIGH — texture is now SLOWER)

`texture_bw.cu`: cycle through array in 32-element steps with ILP, compare `tex1Dfetch<float>` vs `__ldg(p)` vs `data[i]`.

| WS | tex BW | `__ldg` BW | global BW | tex vs ldg |
|---|---|---|---|---|
| L1-resident (16-64 KB) | 8 TB/s | 15 TB/s | 15 TB/s | **2× slower** |
| L2/DRAM | 3 TB/s | 9 TB/s | 9 TB/s | **3× slower** |

**Recommendation:** Use texture only if you need filtering, address-mode clamping/wrap, or normalized-coordinate access. **For raw global fetches `__ldg` (= `ld.global.nc`) is strictly faster on B300.** This reverses Kepler/Maxwell-era guidance.

---

## 9. Surface objects (MED — opposite of texture)

`surface_objects.cu`: 4096×4096 float surface vs same-shape linear memory, repeated reads.

- Surface reads via `surf2Dread`: ~5% **faster** than equivalent linear-memory `data[y*W+x]` reads.
- Why: surfaces use the cudaArray layout (block-linear), giving better 2D spatial locality at the L2/L1 boundary.
- This is the *opposite* of texture's slowdown — `surf2Dread` does not go through the texture filtering hardware.

**Use surfaces when you need a 2D writable cache-friendly buffer; they out-perform `cudaMalloc` for image-like access patterns.**

---

## 10. The "4097 cliff" — NOT a cache effect (HIGH, inv 16)

The `4097-byte cliff` listed in agent assignments is a misnomer. The actual finding is:

**4096 vs 4097 in cuBLAS BF16 GEMM:**
- M=N=K=4096: **1842 TFLOPS** (algo 66, fast Blackwell tensor path)
- M=N=K=4097: **180 TFLOPS** (algo 24 fallback)
- Ratio: **10.2× slower** (NOT 30× as earlier claimed)

This is a **cuBLAS algorithm-selection cliff**, not a cache cliff:
- cuBLAS only offers algo 66 for M and K aligned to multiples of 8 (and tile-divisible).
- Workspace size doesn't help — algo 66 is simply not proposed for misaligned shapes.
- Mitigation: pad M and K to multiple of 32; N is forgiving.

There is **no memory-system cliff at any 4097-byte boundary** that we have evidence for. If "4097-byte cliff" was meant as a separate thing (page boundaries, smem bank striping), no investigation in this repo confirms one.

---

## 11. Latency reference (HIGH)

| Level | Latency |
|---|---|
| Register | 1 cy |
| L1 hit | **42-45 cy** (~22 ns @ 2032 MHz) |
| Shared memory | 24 cy |
| L1→L2 transition | 130-200 cy warm |
| L2 hit | **300-310 cy** (~157 ns @ 1920 MHz) |
| DRAM (cold) | **900-1700 cy** (~470-890 ns) |

L2 latency depends on which side the line sits on (near ~310 cy, far ~660 cy across the 2-partition XBAR); reported number is the average.

---

## 12. Confidence summary

| Claim | Confidence | Verification |
|---|---|---|
| L2 = 126.5 MB | HIGH | `cudaDeviceProp` |
| Max persisting L2 = 79 MB | HIGH | `cudaDeviceGetAttribute` |
| L1+SHMEM = 256 KB unified | HIGH | carveout sweep, attribute API |
| L1 hit = 42-45 cy | HIGH | inv 12 ca-vs-cg cross-check |
| L2 BW @ co=100 = 17 TB/s | HIGH | inv 06 + ncu sector match |
| L2 BW @ co=0 = 22-26 TB/s | MED | catalog only, not re-verified post-firmware |
| L2 BW peak 30+ TB/s @ tiny WS | MED | really L1-amplified, not pure L2 |
| Texture 2-3× slower than `__ldg` | HIGH | texture_bw.cu, both paths SASS-correct |
| Surface ~5% faster than linear | MED | single test, plausible mechanism |
| Persistent L2 zero benefit (hot fits L2) | MED | persistent_l2.cu; theoretical agreement |
| `.cg` 1.25× slower than `.ca` for L2-hot | HIGH | hint-load sweep, ncu-confirmed L2 sector delta |
| 4097 "cliff" = cuBLAS algo selection | HIGH | inv 16 reproduces 10.2×, identifies algos 66/24 |

---

## 13. What changed from earlier docs

- "L2 = 256 MB": wrong, was unit confusion → correct is 126.5 MB.
- "L2 BW 36 TB/s": really LSU/L1-dispatch ceiling at WS<2 MB with `.ca`, not pure L2.
- "L2 BW 10 TB/s": under-occupied launch (148 CTAs × 128 threads = 0.5 warps/SMSP).
- "L1 = 32 KB": was at default carveout (≈100, max SHMEM); true L1 ranges 20-228 KB across carveout.
- "70 MB L2 knee": TLP-hiding artifact, not a capacity wall.
- "30× cliff at 4097": actually 10.2× and is a cuBLAS issue, not memory.
- "Texture is faster for read-only data": **REVERSED on B300** — texture is 2-3× slower.
- "Persistent L2 helps any hot region": false when hot region fits in regular L2.

---

## 14. Open / unresolved

- **Isolated CCTL.IVALL cost** (no PTX exposes it standalone).
- **Per-partition L2 BW under contention** — needs `lts__t_sectors` per-partition breakdown via ncu `fbpa__*` metrics.
- **Carveout=0 vs 100 BW gap on current firmware** — modern repro of the 22 → 17 TB/s drop at co=100 was not done at co=0; the gap may have closed.
- **Best access pattern to maximize L2 hit rate** at WS just below 126 MB — strided patterns show partial misses earlier than expected.
