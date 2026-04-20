# Write Power vs Popcount Density: L2 + DRAM-8G

**Date: 2026-04-20.** Same popcount sweep, but for WRITES instead of reads.
Each store sends a value with EXACTLY `density` bits set in pseudo-random
positions (per (tid, iter) so adjacent writes have different bit positions).

## Setup

- L2 write: `tests/bench_pwr_l2_popcount_write.cu`, .cg store, 64 MB ws.
- DRAM write: `tests/bench_pwr_dram_popcount_write64.cu`, .cg store, 8 GiB ws.
- Each thread precomputes 32 distinct popcount-d values once, then rotates
  through them in the hot loop (avoids re-running the 32-step shuffle
  per store).
- Init zeros the buffer; main kernel does only writes.
- ~5 s sustained per measurement; clock locked 1005 MHz; 5 samples × 0.4s.

## Results — active power (W above 150 W idle)

| density | L1 read | L2 read | DRAM-8G read | L2 write | DRAM-8G write |
|---------|---------|---------|--------------|----------|---------------|
|  0      |  70     |  223    |  397         | 140      | 264           |
|  1      |  76     |  251    |  439         | 153      | 286           |
|  2      |  81     |  271    |  466         | 163      | 302           |
|  4      |  88     |  309    |  516         | 183      | 329           |
|  8      |  97     |  364    |  583         | 213      | 373           |
| 12      | 104     |  395    |  621         | 228      | 396           |
| 16      | 107     |  405    |  637         | 235      | 405   ← peak  |
| 20      | 107     |  403    |  627         | 237      | 403           |
| 24      | 103     |  378    |  595         | 223      | 388           |
| 28      |  96     |  325    |  530         | 198      | 350           |
| 30      |  91     |  290    |  498         | 181      | 325           |
| 31      |  87     |  271    |  473         | 172      | 311           |
| 32      |  82     |  245    |  441         | 159      | 288           |

## Findings

### 1. Same bell-curve shape for writes
At every tier, write power follows the SAME bell curve as reads, peaking
at d=16 (random) and minimum at d=0 / d=32 (constant). The toggle-energy
model applies symmetrically.

### 2. Writes are LOWER power than reads at every tier and density
| tier | d=16 read | d=16 write | write/read |
|------|-----------|------------|------------|
| L2   | 405       | 235        | 0.58       |
| DRAM | 637       | 405        | 0.64       |

L2 writes are ~58% the power of L2 reads at peak. This is partly because
write throughput (~3 TB/s) is ~5× lower than read throughput (~16 TB/s),
so per-second wire activity is much lower.

### 3. Per-byte energy (estimate)
| op           | active W | BW (TB/s) | nJ/byte |
|--------------|----------|-----------|---------|
| L2 read d=16 | 405      | 16        | 25.3    |
| L2 write d=16| 235      | 3         | 78.4    |
| DRAM read    | 637      | 7.5       | 84.9    |
| DRAM write   | 405      | 3.5       | 115.7   |

**Writes burn ~3× more energy per byte than reads at L2.** Likely because
writes traverse more SerDes stages (write-back path through L2 → mesh
back-channel) and require committing data to a multi-port SRAM array.

DRAM write energy is similar to DRAM read (~36% higher). HBM PHY
write energy includes activation + write driver power; reads include
sense-amp + DBI decode.

### 4. Write asymmetry preserved
At d=32 (all ones) - d=0 (all zeros):
- L2 reads: +22 W (245 - 223)
- L2 writes: +19 W (159 - 140)
- DRAM reads: +44 W
- DRAM writes: +24 W

Write side has slightly smaller asymmetry — the bus driver static-power
component is partially decoupled from data direction.

### 5. The 4 × 4 × 4 picture
Combining everything:
- L1 reads (107) ≈ 6× lower than DRAM reads (637) at peak
- L2 writes (235) ≈ 1.7× lower than L2 reads (405)
- DRAM writes (405) ≈ 0.6× DRAM reads
- BW-normalized: writes are ~3× more expensive per byte than reads

## Implications

- **Read-heavy workloads stress the HBM more**. Read-bound LLM inference
  burns 200+ W more than write-bound (which is rare for inference anyway).
- **Write-back caching policies**: pushing dirty data back to DRAM via
  evict is ~64% the per-second cost of an explicit read, but ~36% MORE
  energy per byte. Eviction should be timed when SMs have spare cycles.
- **Adversarial pattern for thermal**: read workload at d=16 gives the
  highest total power (786 W observed sample, 637 W active over idle).

## Confidence

- **HIGH** for the bell-curve shape match between reads and writes at each tier.
- **HIGH** for the 0.58× / 0.64× write/read power ratio at L2 / DRAM.
- **MED** for the per-byte energy estimates (depend on accurate BW peaks).

## What would change conclusions

- Test L1 writes (currently only L1 reads).
- Test mixed read+write workloads at varying ratios.
- Test write barrier (.wb cache hint that forces write-through).
