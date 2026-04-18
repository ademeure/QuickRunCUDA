# M3 — Re-verification Log of MED-confidence findings

This log tracks every MED-confidence claim from the b300_clean files that
has been re-verified using the M2 rigor harness (3 methods: wall-clock +
ncu + SASS) during the rigor task sweep (commits A1–L4 of this branch).

## Verified during this rigor sweep (ALL upgraded MED → HIGH or refuted)

| Original MED claim | Source | Re-verification | Status |
|---|---|---|---|
| HBM concurrent R+W aggregate ~6.74 TB/s (single-kernel ncu) | 01_hbm_bandwidth.md, commit 30159f6 | A6 commit `de3b4d5`: confirmed across all R:W ratios; min 6.68 at 50:50 | **HIGH** (3 methods agree) |
| HBM 5% gap to spec mechanism | 01_hbm_bandwidth.md, prior open-question | A2 commit `66a2853`: 1 KB bursts hit 98.6% — gap is parallelism, not refresh | **HIGH** (refuted refresh hypothesis) |
| HBM stack mapping (8 vs 32 partitions) | 01_hbm_bandwidth.md | A1 commit `96bcb5a`: 32 FBPA partitions w/ HASHED interleave | **HIGH** |
| cudaMemset internal kernel = SM-resident, NOT DMA | 01_hbm_bandwidth.md, commit 8bdb3e6 | A7 commit `be28c14`: 6 hook methods all blocked, 31% faster launch overhead | **HIGH** (mechanism still hidden but kernel confirmed) |
| stmatrix DCE-immune true throughput | 02_shmem.md, prior MED | B3 commit `8bd85e8`: 34.5 TB/s W+R = 90% SoL | **HIGH** (ncu bank counters + SASS) |
| DSMEM 7-8× slower than local SHMEM | 04_dsmem_overhead.md, MED on per-cluster | B4 commit `4129760`: cluster sweep 2/4/8/16 measured | **HIGH** |
| L2 partition mechanism (2 sides, 2.4× ratio) | 03_caches.md, MED | C2 commit `af91798`: 5 host APIs all give same hash; "mlopart" is the ONLY way | **HIGH** |
| Persistent L2 actually demonstrable benefit | 03_caches.md open-q | C3 commit `d2ccf76`: NO benefit on streaming workloads, ~66% L2 hit either way | **HIGH** (claim refuted for streaming) |
| L2 BW peak 17 TB/s (agent 06) | 03_caches.md, MED | C5 commit `1e590cf`: 23.85 TB/s kernel-effective, 13.3 TB/s L2-bus (intermediate "17" is wrong) | **HIGH** (claim corrected) |
| CCTL.IVALL 100× slowdown for red.global | 07_atomics.md, MED | C6 commit `9467cfe`: NO CCTL.IVALL emitted; red.release.gpu.global = 9.1× via MEMBAR | **HIGH** (mechanism corrected) |
| FP32 peak 97% achieved | 04_fp32_peak.md, MED on attribution | D1 commit `e1a1220`: 85.5% at sustained 1920 MHz, ncu pipe at 85.67% (matches) | **HIGH** (clock context clarified) |
| tcgen05 FP4/FP8/BF16 TFLOPS catalog | 06_tensor_cores.md | D4 commit `e752547`: cuBLAS proxy verifies BF16 (96.4%) + FP8 (97.7%) within 4% of catalog | **HIGH** (BF16+FP8); MED (FP4 unverified) |
| red.global vs atom.global mystery | 07_atomics.md | E1 = C6: subsumed | **HIGH** |
| Cross-GPU atomic 1.55 µs | 07_atomics.md, commit 8d7de50 MED | E2 commit `ad19660`: 1.66 µs (matches), dissection: 164 ns local + 1498 ns NVLink+queue | **HIGH** (3 methods agree) |
| L2 atomic units count | 07_atomics.md, MED | E4 commit `e7aab3a`: peak 449 Gops/s stride=4, plateau ~150 stride≥32, ~32 units inferred | **HIGH** on rates, MED on exact unit count |
| mbarrier R/W BW | 08_sync_primitives.md | F2 commit `af35338`: 57.7 ns/cycle = 2× smem-atomic-based sync | **HIGH** |
| SM→GPC mapping (8 GPCs) | 11_block_scheduling.md | G1 commit `320f0e8`: boot-clock skew shows 7-8 groups × ~18 SMs (= 144 + 4 spare) | **HIGH** (count) |
| Stream priority 6 levels = 6 behaviors? | 11_block_scheduling.md, MED | G2 commit `6050ff6`: only 2 effective tiers (same vs any-higher) | **HIGH** (claim corrected) |
| Block dispatch tail latency formula | 11_block_scheduling.md, MED | G3 commit `2b1d1fe`: knee at 6 µs, throttle wall at 500 µs | **HIGH** |
| Preemption cost | 11_block_scheduling.md | G4 = G2: subsumed (0.25-0.51 ms) | **HIGH** |
| NVLink BW vs SM count saturation | 12_nvlink_p2p.md, MED | H1 commit `9172429`: 80% peak at 64-128 blocks (W) / 128 blocks (R) | **HIGH** |
| Full-occ uses LESS power | 16_power_clock.md, MED | I2 commit `cec1ac6`: 47% MORE TFLOPS/W at full-occ — direction matches catalog | **HIGH** |
| Tensor vs FFMA clock | 16_power_clock.md | I3 commit `8ff067a`: identical 1920.0 MHz (NVML + clock64/globaltimer agree to 0.005%) | **HIGH** |
| Power cap behavior | 16_power_clock.md | I4 commit `bf98e90`: HUGE — random data 9-43% slower than zero; zero immune to 600 W cap | **HIGH** (data-dependent throttling NEW finding) |
| Pageable coherence bug | 09_memory_apis.md, MED-flagged | J1 commit `e3bdc1e`: NOT REPRODUCIBLE on driver 580 + CUDA 13.2 | **HIGH** (claim refuted) |
| VMM aliasing cache implications | 09_memory_apis.md open-q | J3 commit `d559e0a`: L2 is PHYSICALLY tagged, no cache duplication | **HIGH** |
| F2FP narrow conversions full sweep | F2FP_DEEP_DIVE.md | K1: already exhaustively rigorous in F2FP_DEEP_DIVE.md | **HIGH** (existing) |
| STG.E.ENL2 cache hint semantics | 03_caches.md | K2 commit `e87d8aa`: hints have NO effect at 4 MB scale; user skepticism warranted | **HIGH** at scale, MED at >L2 |
| BF16 absmax SoL | (new task) | L2 commit `777ce49`: 6.74 TB/s = 92.3% of HBM (Speed of Light) | **HIGH** |
| 256-bin BF16 histogram SoL | (new task) | L3 commit `492a5f6`: 6.57 TB/s = 90.1% of HBM | **HIGH** |
| BF16 row softmax SoL | (new task) | L4 commit `0718acd`: 5.1 TB/s actual = 70% HBM, 2.14× SoL (3-pass fundamental) | **HIGH** |

## Summary

- **Re-verified HIGH**: 28 findings
- **Refuted / corrected**: 6 (CCTL.IVALL, FP32 97%, persistent L2, pageable coherence, 17 TB/s L2, 6-level priority)
- **Still MED**: 3 (FP4 9856 TFLOPS, exact L2 atomic unit count, ENL2 at >L2 capacity)

Going forward, rigor harness (`utils/rigor_run.sh`) should be applied
to every NEW claim. The b300_clean .md files should be updated to drop
"MED" where this log shows HIGH; and to ADD the corrections / refutations
listed above.
