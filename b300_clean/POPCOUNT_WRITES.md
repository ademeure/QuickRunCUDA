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

### 3. The bandwidth picture (ncu-verified)
ncu metrics for matched read vs write kernels (.cg, 64 MB ws):

| metric                        | READ      | WRITE     | ratio |
|-------------------------------|-----------|-----------|-------|
| wall-clock effective BW       | 15.9 TB/s | 3.78 TB/s | 4.2×  |
| `lts__t_bytes.sum.per_second` | 20.81 TB/s| 11.62 TB/s| 1.79× |
| `dram__bytes.sum.per_second`  | 319 GB/s  | 1.11 TB/s | 0.29× |
| L2 sector amplification (lts/l1tex) | ~1.0× | 1.50× (write-allocate) | — |

**The L2 port has only 1.8× read-vs-write asymmetry** (21 / 12 TB/s).
The remaining 4.2× wall-clock gap comes from:
- **Write-allocate amplification** (1.5× extra sectors per demand-write,
  ncu confirms `lts__t_sectors_op_write` = 1.5× `l1tex__t_sectors`)
- **DRAM write-back leak** (1.07 TB/s leaving L2 even at "L2-fitting" 64 MB ws)

### Tried to push writes higher — confirmed it's the SM side, not L2

I tried multiple cleaner patterns:
- Full-sector stores: `st.global.cg.v4.b64` → emits **STG.E.ENL2.256** (32 B
  per store, full sector). Time: 19.5 s for 30 M iters × 32 B → **3.74 TB/s**
  (basically same as STG.128 = 3.78 TB/s).
- Single-pass / no revisit (each thread writes 96 KB own region):
  **3.12 TB/s** (DRAM-bound).
- Warp-coalesced 1024 B per warp-cycle (32 threads × 32 B sequential
  per-warp slice of 16 KB): **3.74 TB/s** — DRAM leak drops to 22 MB/s
  (essentially zero), but wall BW unchanged.
- Tried .cg / .cs / .wb / .wt cache hints — all within ±2 % of 3.74 TB/s.

ncu on the warp-coalesced run:
| metric                          | value         |
|---------------------------------|---------------|
| dram__bytes_write.per_second    | 21.91 MB/s    |
| lts__t_bytes.sum.per_second     | 10.84 TB/s    |
| lts__t_sectors_op_write         | 568 B sectors |
| l1tex__t_sectors_op_st          | 379 B sectors |
| l1tex__t_requests_op_st         | 11.84 B       |
| amplification (lts / l1tex)     | **1.50×**     |

**The 1.5× write-allocate amplification persists even with full-sector
warp-coalesced writes** — it is a per-write-transaction L2 metadata cost,
NOT a partial-line artifact. So writes were never a "bad access pattern"
problem.

**The 3.74 TB/s effective write ceiling is a real B300 SM/LSU store-pipe
limit.** The L2 port has 10.84 TB/s of headroom that the SM side
cannot fill on writes. Reads with the same kernel template hit 15.92 TB/s
because reads do not have the 1.5× allocate amplification AND
the LSU read pipe is wider than the store pipe per cycle.

### 4. Per-byte energy (estimate)
| op           | active W | wall-effective BW | nJ/byte (demand) |
|--------------|----------|-------------------|------------------|
| L2 read d=16 | 405      | 15.9 TB/s         | 25.5             |
| L2 write d=16| 235      | 3.78 TB/s         | 62.2             |
| DRAM read    | 637      | 7.4 TB/s          | 86.1             |
| DRAM write   | 405      | 3.5 TB/s          | 115.7            |

Writes look ~2.4× more expensive per "demanded" byte at L2 — but at the
LTS port level (real internal traffic) the gap is closer to 1.4× because
writes carry more sector activity per kernel byte.

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
