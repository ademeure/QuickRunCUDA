# M16: V9 Full Synthesis — Latency Ladder, Myth-Busts, Pipeline Model

19 rigor-verified V9 findings. This synthesis supersedes M15 with full coverage.

## I. Compute pipeline ownership map (Blackwell)

| Pipe        | Ops                                          | Solo peak achieved   |
|-------------|----------------------------------------------|----------------------|
| **FMA**     | FFMA, FADD, FMUL, IMAD, DFMA, HMMA          | 75.2 TFLOPS @ 97.6%  |
| **ALU**     | IADD3, LOP3.LUT (XOR/AND fused), SEL, ISETP | ~38 TOPS @ 99.9%     |
| **XU**      | MUFU (rsqrt, sin, exp, log)                  | 47.8 GMUFU/s @ 99.5% |
| **MIO**     | SHFL, LDGSTS queue                           | 3 G warp-SHFL/s      |
| **LSU**     | LDG, STG, LDS, STS, atomic                   | 26.9 TB/s SMEM       |
| **Tensor**  | HMMA, mma.sync legacy                         | 578.6 TFLOPS @ 99.9% |

## II. Single-instruction latency ladder

| Op / primitive                  | Latency (cy) | ns @ 2.032 GHz |
|---------------------------------|--------------|-----------------|
| Register MOV                    | ~1           | 0.5             |
| FFMA / FADD / FMUL              | 4.22         | 2.1             |
| IMAD                            | 4.25         | 2.1             |
| HMMA.F16 (m16n8k16)             | **20**       | 9.8             |
| `__syncwarp` / `fence_block`    | 23           | 11              |
| SMEM LDS                        | 29           | 14              |
| `__syncthreads(4 warps)`        | 30           | 15              |
| L1 hit                          | 47           | 23              |
| DFMA                            | **63.7**     | 31              |
| `__syncthreads(32 warps)`       | 86           | 42              |
| nanosleep(1000)                 | 2066         | 1000            |
| L2 hit                          | ~300         | 148             |
| DRAM (random)                   | ~317         | 156             |
| **fence (GPU)**                 | **281**      | 138             |
| barrier.cluster                 | 370          | 182             |
| **global atomic (chained)**     | **697**      | 343             |
| **fence_system**                | **3019**     | 1486            |

Formula: `__syncthreads() = 22 + 2×N_warps cy` (HIGH conf, exact fit)

## III. Myth-busts caught by rigor (Rule 9)

5 false claims caught & corrected via 10-rule protocol:

1. **"Atomic scope gives 17× speedup"** → FALSE.
   Original test was apples-oranges (varying address vs fixed). Corrected:
   all scopes have same chained latency 697 cy; scope only affects
   coherence, not single-thread latency.

2. **"2-way branch divergence is 1.09× free"** → MISLEADING.
   Original test used switch-case with constants → compiler PREDICATED.
   True 2-way divergence (FFMA vs MUFU): 2.57× slower.

3. **"Mixed FMA + ALU = 114 TOPS combined"** → FALSE.
   Hypothetical sum of peaks. Empirical: 8/8 mix peaks at 131% pipe sum
   but ~74 TOPS combined throughput (similar to solo FMA peak).

4. **"cudaGraph reduces per-kernel launch latency"** → FALSE.
   Single-kernel graph = 2.05 µs = SAME as direct. Speedup only from
   batching N kernels per graph (3.84× at N=100).

5. **"nanosleep with __syncwarp takes MAX"** → FALSE.
   Divergent N values per lane → warp sleeps ~MIN, not MAX.

## IV. Async load BW finding (V9 cp.async)

| Path                  | BW        | % of HBM peak |
|-----------------------|-----------|---------------|
| Plain LDG coalesced   | 5.82 TB/s | 81%           |
| **cp.async.ca**       | **6.91 TB/s** | **96%** ← async wins |
| TMA bulk store        | 7.57 TB/s | 95%           |

cp.async beats LDG by 19% — bypasses RF + L1 bank pressure. Why cuBLAS uses async loads.

## V. Per-warp dispatch & pipeline model

- 4 SMSPs per SM, each issues 1 instruction per cycle when warps available
- Per warp: 1 instruction per cycle (no double-issue)
- Pipeline depth: 4 cy for FMA (need 4 chains to hide latency)
- Pipeline depth: 64 cy for DFMA (single FP64 port; 1 chain saturates)
- Pipeline depth: 20 cy for HMMA (need 5 chains; 8 gives 99.9%)

## VI. Critical kernel design rules

1. **Register budget**: keep live vars < (65536 / occupancy_thr) or hit 9× spill cliff
2. **Block size**: 128 thr (4 warps) optimal — minimum sync cost (30 cy) + good occupancy
3. **Branch divergence**: simple 2-way predicates fine; >4-way truly serializes
4. **Atomic scope**: irrelevant for single-thread latency; matters for parallel
5. **Fence scope**: use finest needed (block free, GPU 138 ns, system 1.5 µs)
6. **nanosleep**: ALL lanes must use SAME N (divergent → MIN behavior)
7. **cudaGraph**: only useful for batching N kernels, not individual launches
8. **Subnormals**: free with `-use_fast_math` (FTZ)

## VII. Cross-pipeline parallelism (corrected)

Mixing FMA + ALU work:
- Both pipes ARE simultaneously active (pipe sum 131% peak)
- BUT per-warp issue rate of 1 inst/cy prevents stacking peaks
- Actual combined throughput plateaus at ~75 TOPS (similar to solo FMA peak)
- Useful when both kinds of work needed (GEMM addressing + compute), not as 2× hack

## VIII. V9 confidence summary

19 items, all 10-rule rigor:
- HIGH (15): cp.async, op latency, mem latency, syncthreads, HMMA latency, LDS,
  syncwarp, atomic chained, atomic pipelined, fence variants, regspill,
  subnormal, int op pipes, mixed pipes correction, cudaGraph correction
- MEDIUM (4): nanosleep rounding, branch divergence, mem latency L2/DRAM
  distinction, nanosleep threads exact mechanism

## V9 → V10 transition

V9 essentially complete. Remaining gaps for V10:
- TMA bulk load BW (mbarrier completion required)
- mbarrier latency
- HMMA.SP sparse latency
- ldmatrix with correct swizzle
- L1 cache associativity / replacement
- Voltage curve (DVS scaling exact)
- tcgen05.mma working (cuTLASS UmmaDesc reference)