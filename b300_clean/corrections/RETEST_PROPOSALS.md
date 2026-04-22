# RETEST_PROPOSALS — Concrete .cu Sketches for the 5 Highest-Impact Unresolved Items

Author: f2fp-deep-dive doubt swarm, 2026-04-22.
Reference style: `tests/standalone/v46_tma_inflight.cu`, `v49_dual_pipe.cu`, `v50_warp_specialized.cu`, `v51_multistream_hbm.cu`.
Build: standalone `.cu` files compiled with `nvcc -arch=sm_103a -O3 -std=c++17 <file>.cu -o <name>` (no QuickRunCUDA harness needed — these are self-contained).
Conventions: BLOCKS=148 (1×SM) or 1184 (8×SM) for occupancy variants. RUNS≥3 with cudaEvent. All tests must be `clock64`-bracketed AND event-bracketed for cross-validation. Always `pkill -9 QuickRunCUDA && sleep 5` between runs. `nvidia-smi -lgc 1920` for locked-clock claims, no lock for boost-clock claims — STATE which.

---

## SKETCH 1 — `v52_dual_issue_warp_sweep.cu`

**Settles:** UNRESOLVED #1 — Is V49/V50's 55%/74% dual-pipe "ceiling" a real architectural cap or just under-occupancy?

**Hypothesis matrix:**
- H_arch: dispatch is genuinely capped near 1 inst/SMSP/cy → adding more warps does nothing past ≈2 warps/SMSP.
- H_occ: more warps mask back-to-back FFMA latency → throughput climbs with warps/SMSP up to 4 or 8.

**Kernel:**

```cpp
// V52: Dual-issue warps-per-SMSP sweep.
// V49 OP=2 (FFMA + LOP3 same warp) capped at 55%. Sweep occupancy: 1/2/4/8 warps/SMSP.
// If throughput plateaus by 2 warps/SMSP → architectural dispatch cap. If climbs to 8 → was occupancy-bound.

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

// Unified template — vary THREADS via launch_bounds and BLOCKS_PER_SM via second arg.
// 4 SMSPs/SM. warps_per_SMSP = (THREADS/32) * BLOCKS_PER_SM / 4.
template<int THREADS, int BLOCKS_PER_SM, int ILP, int N_ITERS>
__global__ __launch_bounds__(THREADS, BLOCKS_PER_SM)
void v52_kernel(unsigned* out) {
    int tid = threadIdx.x;
    float f[8]; unsigned u[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) { f[k] = (float)(tid + k); u[k] = tid * 7 + k * 13; }

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // V49 OP=2 body verbatim — interleaved FFMA + LOP3, same warp.
    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            asm volatile("fma.rn.f32 %0, %0, 0f3FC00000, 0f3F000000;" : "+f"(f[k]));
            asm volatile("lop3.b32 %0, %0, 0xa5a5a5a5, 0x12345678, 0x96;" : "+r"(u[k]));
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) {
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= u[k] ^ __float_as_int(f[k]);
        out[blockIdx.x] = acc + (unsigned)(t1 - t0);
    }
}

int main() {
    CK(cudaSetDevice(0));
    unsigned* d_out; CK(cudaMalloc(&d_out, 1184 * 4));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    const int N_ITERS = 5000, ILP = 8, BLOCKS = 1184;  // 8 CTAs/SM ceiling

    // (warps_per_CTA, blocks_per_SM, total_warps_per_SMSP)
    struct Cfg { int t, bps; const char* name; };
    Cfg cfgs[] = {
        { 32, 1, "1 warp / 1 CTA   = 0.25 warps/SMSP" },  // unrealistic; below 1
        { 32, 4, "1 warp / 4 CTA   = 1   warps/SMSP" },
        { 64, 4, "2 warp / 4 CTA   = 2   warps/SMSP" },   // V49 baseline
        {128, 4, "4 warp / 4 CTA   = 4   warps/SMSP" },
        {256, 2, "8 warp / 2 CTA   = 4   warps/SMSP (alt)" },
        {256, 4, "8 warp / 4 CTA   = 8   warps/SMSP" },
        {512, 2, "16 warp / 2 CTA  = 8   warps/SMSP (alt)" },
        {512, 4, "16 warp / 4 CTA  = 16  warps/SMSP" },   // RF cap may force eviction
    };

    printf("=== V52 Warps-per-SMSP sweep on V49 OP=2 (FFMA+LOP3 dual) ===\n");
    printf("threads bps  warps/SMSP  wall_ms  Glane/s   ratio_to_baseline\n");
    double base = 0;
    for (auto& c : cfgs) {
        // Dispatch via switch — templates can't be runtime-selected.
        auto run = [&](){
            #define LAUNCH(T,B) v52_kernel<T,B,ILP,N_ITERS><<<BLOCKS,T>>>(d_out)
            if (c.t==32 && c.bps==1) LAUNCH(32,1);
            else if (c.t==32 && c.bps==4) LAUNCH(32,4);
            else if (c.t==64 && c.bps==4) LAUNCH(64,4);
            else if (c.t==128 && c.bps==4) LAUNCH(128,4);
            else if (c.t==256 && c.bps==2) LAUNCH(256,2);
            else if (c.t==256 && c.bps==4) LAUNCH(256,4);
            else if (c.t==512 && c.bps==2) LAUNCH(512,2);
            else if (c.t==512 && c.bps==4) LAUNCH(512,4);
            #undef LAUNCH
        };
        run(); cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("%s -- LAUNCH FAIL (RF?)\n", c.name); cudaGetLastError(); continue; }

        float total_ms = 0;
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(e0); run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); total_ms += ms;
        }
        float ms = total_ms / 5.0f;
        // Each thread issues ILP × 2 ops (FFMA + LOP3) × N_ITERS
        double total = (double)BLOCKS * c.t * ILP * 2 * N_ITERS;
        double glane = total / (ms / 1e3) / 1e9;
        if (base == 0) base = glane;
        printf("%s  %.2f  %.1f  %.2fx\n", c.name, ms, glane, glane/base);
    }
    return 0;
}
```

**ncu metrics to collect** (per-config, run with `ncu --metrics ...`):

```
smsp__inst_issued.sum.per_cycle_active           # avg insts dispatched per SMSP-cy
smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active   # FFMA pipe util
smsp__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active   # LOP3/INT pipe util
smsp__warps_active.avg.pct_of_peak_sustained_active             # occupancy actual
smsp__cycles_active.avg
sm__warps_active.avg.per_cycle_active
```

**Predicted outcome under each hypothesis:**

| warps/SMSP | H_arch (dispatch capped) | H_occ (latency-bound) |
|---|---|---|
| 1 | ~50% baseline (low) | low |
| 2 | 55% (V49) | growing |
| 4 | 55% (FLAT) | ~75% |
| 8 | 55% (FLAT) | ~95-100% |
| 16 | 55% (or RF spill) | ~100% (or RF spill) |

**Decision rule:**
- If `inst_issued.per_cycle_active` saturates ≤ 1.05 across all warps/SMSP → **dispatch cap is REAL** (architectural; matches the ALU/FMA shared-scheduler hypothesis from V49).
- If `pipe_fma_cycles_active + pipe_alu_cycles_active` sum > 130% at warps≥4 → **dispatch cap is FALSE**; pipes ARE separate, V49 was occupancy-starved.
- If sum stays at ~100% but each pipe individually grows → **dispatch slot is the cap, not the pipes** (pipes share dispatch port).

---

## SKETCH 2 — `v53_dsmem_fenced_retest.cu`

**Settles:** UNRESOLVED #2 — Is V21's DSMEM `push_ring_wr` measurement a true read/write SoL or did missing fences let stores go in-flight at clock64?

**Two-axis design:** (a) write side with explicit `fence.sc.cluster` between every store batch and the closing `clock64`; (b) read side with ILP loop where the next read's address is loop-carried but NOT result-dependent.

**Kernel:**

```cpp
// V53: DSMEM read/write SoL with proper fences.
// V21 timed pushes without fence -> stores still in flight at clock64 stop.
// Add fence.sc.cluster after every store batch. Separate non-chained ILP read.

#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

// Cluster of CLUSTER_SIZE CTAs share DSMEM. Each CTA writes ILP doublewords/iter into peer's smem.
template<int CLUSTER_SIZE, int ILP, int N_ITERS, int FENCED>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1) __launch_bounds__(128, 1)
void v53_dsmem_write(unsigned* out) {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    __shared__ __align__(16) unsigned smem[1024];

    int tid = threadIdx.x;
    int my_rank = cluster.block_rank();
    int peer = (my_rank + 1) % CLUSTER_SIZE;
    unsigned* peer_smem = cluster.map_shared_rank(smem, peer);

    cluster.sync();

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            // Distributed shared store: st.shared::cluster.u32
            unsigned val = it * 17 + tid * 31 + k;
            unsigned addr = (unsigned)__cvta_generic_to_shared(&peer_smem[(tid + k * 32) & 1023]);
            asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(addr), "r"(val) : "memory");
        }
        if (FENCED) {
            // CRITICAL: ensure every store completes before next clock64 tick is meaningful.
            asm volatile("fence.sc.cluster;" ::: "memory");
        }
    }
    if (FENCED) asm volatile("fence.sc.cluster;" ::: "memory");

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    cluster.sync();
    if (tid == 0) out[blockIdx.x] = smem[0] + (unsigned)(t1 - t0);
}

// Read SoL with non-chained ILP — addresses are loop-carried but values are NOT.
template<int CLUSTER_SIZE, int ILP, int N_ITERS>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1) __launch_bounds__(128, 1)
void v53_dsmem_read(unsigned* out) {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    __shared__ __align__(16) unsigned smem[1024];

    int tid = threadIdx.x;
    int my_rank = cluster.block_rank();
    int peer = (my_rank + 1) % CLUSTER_SIZE;
    unsigned* peer_smem = cluster.map_shared_rank(smem, peer);
    smem[tid] = tid;
    cluster.sync();

    // Address chain: next address depends on loop variable, NOT loaded value.
    // Values (vals[]) accumulate via XOR but are not in the address path → no dep chain.
    unsigned vals[8] = {0};
    unsigned addrs[8];
    #pragma unroll
    for (int k = 0; k < ILP; k++)
        addrs[k] = (unsigned)__cvta_generic_to_shared(&peer_smem[(tid + k * 32) & 1023]);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            unsigned v;
            asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(addrs[k]));
            vals[k] ^= v;     // not in addr path
        }
        // address rotation is loop-carried (cheap ALU) but NOT result-dependent
        #pragma unroll
        for (int k = 0; k < ILP; k++) addrs[k] += 4;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    cluster.sync();
    if (tid == 0) {
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= vals[k];
        out[blockIdx.x] = acc + (unsigned)(t1 - t0);
    }
}

int main() {
    CK(cudaSetDevice(0));
    unsigned* d_out; CK(cudaMalloc(&d_out, 4096));
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    const int N_ITERS = 8192, ILP = 8, CLUSTER = 8, BLOCKS = 1184; // 1184/8 = 148 clusters

    printf("=== V53 DSMEM read/write SoL with fence.sc.cluster ===\n");

    auto bench = [&](const char* lbl, auto kern, double bytes_per_op) {
        kern<<<BLOCKS, 128>>>(d_out); cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("%s FAIL\n", lbl); cudaGetLastError(); return; }
        float total = 0;
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(e0); kern<<<BLOCKS, 128>>>(d_out); cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); total += ms;
        }
        float ms = total/5;
        double ops = (double)BLOCKS * 128 * ILP * N_ITERS;
        double tbs = ops * bytes_per_op / (ms/1e3) / 1e12;
        printf("%-30s ms=%.3f ops=%.2eG TB/s=%.3f\n", lbl, ms, ops/1e9, tbs);
    };

    bench("WR no-fence (V21 mode)",  v53_dsmem_write<CLUSTER,ILP,N_ITERS,0>, 4.0);
    bench("WR fence.sc.cluster",     v53_dsmem_write<CLUSTER,ILP,N_ITERS,1>, 4.0);
    bench("RD non-chained ILP",      v53_dsmem_read<CLUSTER,ILP,N_ITERS>,    4.0);

    return 0;
}
```

**ncu metrics:**
```
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
sm__inst_executed_pipe_lsu.sum
smsp__inst_executed_op_st_shared.sum
smsp__inst_executed_op_ld_shared.sum
sm__cycles_elapsed.avg
```

**Predicted outcome:**
| Test | If V21 was correct | If V21 was missing-fence artifact |
|---|---|---|
| WR no-fence | Same as V21 | Same as V21 (high) |
| WR fenced | Same as V21 | **Significantly slower** (true latency surfaces) |
| RD non-chained | Same as RD-chained V21 | **Higher** than V21 (no dep chain) |

**Decision rule:**
- If `WR_fenced / WR_unfenced` ratio > 1.3 → V21's write SoL was inflated; reduce to fenced number.
- If ratio < 1.05 → V21 measurement holds.
- If `RD non-chained > RD V21` by >1.2× → DSMEM read SoL needs upgrading.

---

## SKETCH 3 — `v54_membar_isolation.cu`

**Settles:** UNRESOLVED #3 — Threadfence_system 1750/2870/3042 cy spread (1.74×). Establish authoritative number with N-issue scaling.

**Kernel:**

```cpp
// V54: membar.{cta,gpu,sys} latency, single-thread, single-issue baseline + N-issue scaling.
// Uses fence.acq_rel as inert barrier marker so clock64 deltas frame exactly the membar.
// SASS-verify after compile: nvdisasm -c output.cubin | grep MEMBAR

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int SCOPE, int N_ISSUE>  // SCOPE: 0=cta 1=gpu 2=sys
__global__ __launch_bounds__(32, 1)
void v54_membar(unsigned long long* out, volatile unsigned* probe) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Touch global so the prior store has something coherent to flush.
    probe[0] = 0xdeadbeef;

    // Inert acq_rel marker (no fabric round trip on its own, but blocks reordering).
    asm volatile("fence.acq_rel.gpu;" ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll
    for (int i = 0; i < N_ISSUE; i++) {
        if (SCOPE == 0)      asm volatile("membar.cta;" ::: "memory");
        else if (SCOPE == 1) asm volatile("membar.gl;"  ::: "memory");  // .gl == GPU
        else                 asm volatile("membar.sys;" ::: "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    out[SCOPE * 16 + N_ISSUE] = t1 - t0;
}

int main() {
    CK(cudaSetDevice(0));
    unsigned long long* d_out; CK(cudaMalloc(&d_out, 4096));
    unsigned* d_probe; CK(cudaMalloc(&d_probe, 4));
    cudaMemset(d_out, 0, 4096);

    printf("=== V54 membar isolation (1-warp, 1-thread) ===\n");
    printf("scope     N=1     N=2     N=4     N=8    cy/issue (avg)\n");

    const char* names[] = {"membar.cta", "membar.gl ", "membar.sys"};

    auto launch = [&](int sc, int ni) {
        if (sc == 0) {
            if      (ni == 1) v54_membar<0,1><<<1,32>>>(d_out, d_probe);
            else if (ni == 2) v54_membar<0,2><<<1,32>>>(d_out, d_probe);
            else if (ni == 4) v54_membar<0,4><<<1,32>>>(d_out, d_probe);
            else if (ni == 8) v54_membar<0,8><<<1,32>>>(d_out, d_probe);
        } else if (sc == 1) {
            if      (ni == 1) v54_membar<1,1><<<1,32>>>(d_out, d_probe);
            else if (ni == 2) v54_membar<1,2><<<1,32>>>(d_out, d_probe);
            else if (ni == 4) v54_membar<1,4><<<1,32>>>(d_out, d_probe);
            else if (ni == 8) v54_membar<1,8><<<1,32>>>(d_out, d_probe);
        } else {
            if      (ni == 1) v54_membar<2,1><<<1,32>>>(d_out, d_probe);
            else if (ni == 2) v54_membar<2,2><<<1,32>>>(d_out, d_probe);
            else if (ni == 4) v54_membar<2,4><<<1,32>>>(d_out, d_probe);
            else if (ni == 8) v54_membar<2,8><<<1,32>>>(d_out, d_probe);
        }
    };

    for (int sc = 0; sc < 3; sc++) {
        unsigned long long cy[5] = {0};
        for (int idx = 0; idx < 4; idx++) {
            int ni = 1 << idx;
            // Median of 21 runs
            unsigned long long samples[21];
            for (int s = 0; s < 21; s++) {
                launch(sc, ni); cudaDeviceSynchronize();
                cudaMemcpy(&samples[s], d_out + sc*16 + ni, 8, cudaMemcpyDeviceToHost);
            }
            // Bubble sort (small)
            for (int a=0;a<21;a++) for (int b=a+1;b<21;b++) if (samples[b]<samples[a]) { auto t=samples[a]; samples[a]=samples[b]; samples[b]=t; }
            cy[idx] = samples[10];
        }
        double per_issue = (double)(cy[3] - cy[0]) / (8 - 1);  // slope, robust to fixed clock64 overhead
        printf("%s  %4llu    %4llu    %4llu    %4llu    %.1f\n",
               names[sc], cy[0], cy[1], cy[2], cy[3], per_issue);
    }

    // SASS verification
    printf("\nSASS-verify: nvdisasm output should show exactly one MEMBAR.{CTA,GL,SYS} per inline asm.\n");
    return 0;
}
```

**ncu metrics:**
```
sm__cycles_elapsed.avg                          # cross-check clock64 base
smsp__inst_executed_op_membar.sum               # confirm count
sm__warps_active.avg.per_cycle_active           # should be ~1/SM (single warp)
```
Plus offline: `cuobjdump --dump-sass v54_membar` and grep for `MEMBAR`.

**Predicted outcome / decision rule:**
- If per-issue slope for `membar.sys` is ~280-320 cy → matches `fence.sc.sys` 2870/8 ≈ 320 (DSMEM-style amortized issue). **Then 1750 cy was a 6-deep amortized batch and 3042 was over-counting setup.**
- If single-issue (N=1) ~1750 cy and 8-issue per-slope ~250 cy → there is fixed setup ~1500 cy + ~250/issue. Both prior numbers were partial truths.
- If N=1 ≈ 3042 cy → V9 number wins; 08's 1750 was undercount.
- If `membar.cta` differs by >2× from F6's 6 cy → revisit F6 too.

Cement choice: report **median single-issue at locked 1920 MHz**, with full N=1..8 table appended.

---

## SKETCH 4 — `v55_hbm_floor_BEST.cu`

**Settles:** UNRESOLVED #4 — Anchor "% of HBM peak" denominator with the BEST-known recipe.

**Recipe (from V32, V46, V48 lessons):**
- TMA bulk loads (`cp.async.bulk.shared::cluster.global`)
- 16 KB tile, 8-deep in-flight per CTA
- 148 CTAs (1×SM), per-warp issue (4 issuer warps × 2 inflight = 8 inflight/CTA)
- Working set = 4 GB so L2 (126 MB) hit rate ≈ 0
- N_ITERS chosen to give ≥10 ms wall (anti-launch-overhead)
- Sweep tile size {4, 8, 16, 32, 64} KB to find sweet spot.

**Kernel** (extends v46 with multi-warp issuers):

```cpp
// V55: HBM3E empirical floor — best-known recipe.
// Goal: maximum sustained HBM read BW. Used to anchor "% of peak" denominator.

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int TILE_BYTES, int N_INFLIGHT, int N_ITERS, int N_ISSUE_WARPS>
__global__ __launch_bounds__(128, 1)
void v55_hbm_best(const float* src, unsigned long long* out, unsigned total_ctas, size_t cap_words) {
    extern __shared__ __align__(16) char buf_raw[];
    __shared__ __align__(8) unsigned long long mbar[16];

    int tid = threadIdx.x;
    int wid = tid / 32;
    int bid = blockIdx.x;

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < N_INFLIGHT; i++)
            asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                :: "r"((unsigned)__cvta_generic_to_shared(&mbar[i])) : "memory");
    }
    __syncthreads();

    unsigned bufs[16], mbars[16];
    #pragma unroll
    for (int i = 0; i < N_INFLIGHT; i++) {
        bufs[i]  = (unsigned)__cvta_generic_to_shared(&buf_raw[i * TILE_BYTES]);
        mbars[i] = (unsigned)__cvta_generic_to_shared(&mbar[i]);
    }

    // Strided over 4 GB with bid-bias to defeat L2.
    size_t stride = total_ctas * (TILE_BYTES / 4);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // PER-WARP issue: 4 warps, each owns N_INFLIGHT/N_ISSUE_WARPS slots.
    int slots_per_warp = N_INFLIGHT / N_ISSUE_WARPS;
    int slot_base = wid * slots_per_warp;

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        if (tid % 32 == 0 && wid < N_ISSUE_WARPS) {
            #pragma unroll
            for (int s = 0; s < slots_per_warp; s++) {
                int i = slot_base + s;
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                    :: "r"(mbars[i]), "r"(TILE_BYTES) : "memory");
                size_t off = (bid * (size_t)(TILE_BYTES/4)
                            + (size_t)(it * N_INFLIGHT + i) * stride) % (cap_words - TILE_BYTES/4);
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];"
                    :: "r"(bufs[i]), "l"(src + off), "r"(TILE_BYTES), "r"(mbars[i])
                    : "memory");
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int i = 0; i < N_INFLIGHT; i++) {
                int done = 0; int spin = 0;
                while (!done && spin < 1000000) {
                    asm volatile("{.reg .pred p; mbarrier.try_wait.shared.b64 p, [%1], 0; selp.u32 %0,1,0,p;}"
                        : "=r"(done) : "r"(mbars[i]) : "memory");
                    spin++;
                }
            }
        }
        __syncthreads();
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (tid == 0 && bid == 0) {
        out[0] = t1 - t0;
        ((float*)&out[2])[0] = ((float*)buf_raw)[0];
    }
}

int main() {
    CK(cudaSetDevice(0));
    size_t words = 1ull << 30;  // 4 GB
    float* d_src; CK(cudaMalloc(&d_src, words * 4));
    cudaMemset(d_src, 0xa5, words * 4);
    unsigned long long* d_out; CK(cudaMalloc(&d_out, 256));
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("=== V55 HBM3E empirical floor (best recipe) ===\n");
    printf("Spec: 8 TB/s. Prior best: 7.31 TB/s (V32). Looking for ceiling.\n\n");
    printf("tile_KB inflight issue_warps shmem_KB N_iters wall_ms TB/s pct_of_8\n");

    #define TRY(TILE, NI, NW, ITS) do {                                        \
        cudaFuncSetAttribute(v55_hbm_best<TILE,NI,ITS,NW>,                     \
            cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024);            \
        int shmem = NI * TILE; if (shmem > 200*1024) break;                    \
        v55_hbm_best<TILE,NI,ITS,NW><<<148,128,shmem>>>((const float*)d_src,d_out,148,words); \
        cudaDeviceSynchronize(); if (cudaGetLastError()) { printf("%d/%d/%d FAIL\n",TILE,NI,NW); cudaGetLastError(); break; } \
        float total = 0;                                                       \
        for (int r=0;r<5;r++) { cudaEventRecord(e0);                           \
            v55_hbm_best<TILE,NI,ITS,NW><<<148,128,shmem>>>((const float*)d_src,d_out,148,words); \
            cudaEventRecord(e1); cudaEventSynchronize(e1);                     \
            float ms; cudaEventElapsedTime(&ms,e0,e1); total+=ms; }            \
        float ms = total/5; double bytes = (double)148*ITS*NI*TILE;            \
        double tbs = bytes/(ms/1e3)/1e12;                                      \
        printf("%5d   %4d     %3d         %5d   %4d   %.3f  %.3f  %.1f%%\n",   \
            TILE/1024,NI,NW,shmem/1024,ITS,ms,tbs,tbs/8.0*100);                \
    } while(0)

    // Sweep tile size at fixed depth
    TRY( 4096, 8, 4, 1024);
    TRY( 8192, 8, 4,  512);
    TRY(16384, 8, 4,  256);  // V46 baseline
    TRY(32768, 8, 4,  128);
    TRY(65536, 4, 4,   64);
    // Sweep depth at best tile
    TRY(16384, 4, 2,  256);
    TRY(16384, 8, 2,  256);
    TRY(16384, 8, 4,  256);
    // Persistent-equivalent (1×SM) but more iters to ensure ≥20 ms
    TRY(16384, 8, 4, 1024);

    return 0;
}
```

**ncu metrics:**
```
dram__bytes_read.sum.per_second                  # authoritative HBM read BW
lts__t_sectors_op_read.sum.pct_of_peak_sustained # L2 traffic (should be ~0)
lts__t_sector_hit_rate.pct                       # confirm L2 hit rate < 5%
sm__warps_active.avg.pct_of_peak_sustained
```

**Decision rule:**
- Take MAX TB/s across the sweep where `lts__t_sector_hit_rate.pct < 10%` → empirical HBM floor.
- Compare to V32's 7.31 TB/s and to 8 TB/s spec. Use this number as the denominator for ALL "% of HBM peak" claims going forward.
- If the empirical floor is e.g. 7.5 TB/s, retroactively rescale claims that used 8.0 TB/s denominator.

---

## SKETCH 5 — `v56_nvfp4_AB_mechanism.cu`

**Settles:** UNRESOLVED #5 — Why is power so asymmetric A>>B vs B>>A in NVFP4 K=96 tcgen05.mma? 4 candidate mechanisms.

**Candidates:**
1. **TMA multicast** — A is multicast to N CTAs in cluster, B is unicast. Different bus.
2. **A↔B operand swap** — internally MMA treats A and B differently in the SMEM staging.
3. **SMEM dwell time** — A sits longer in SMEM (reused across K loop), B refreshed each step.
4. **Pipeline depth** — A side has deeper double-buffer than B.

**Kernel sketch** (one driver, four mode toggles):

```cpp
// V56: NVFP4 A vs B mechanism discriminator (tcgen05.mma).
// Run 4 controlled tilt tests; each ELIMINATES one candidate.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

// We assume a working tcgen05.mma harness exists (e.g., based on tests/bench_nvfp4_*.cu).
// MODE encodes the experimental knob.
//   MODE=0 baseline (A toggling, B static)
//   MODE=1 swap roles (A static, B toggling)            -> tests #2 swap
//   MODE=2 use cluster MULTICAST for B too              -> tests #1 multicast asymmetry
//   MODE=3 single-buffered A (no double-buffer reuse)   -> tests #3/#4 dwell/depth
//   MODE=4 cluster size 1 (no multicast at all)         -> tests #1 again
template<int MODE, int CLUSTER, int K_DEPTH>
__global__ __cluster_dims__(CLUSTER,1,1) __launch_bounds__(128,1)
void v56_nvfp4_mma(const uint4* A_data, const uint4* B_data, uint32_t* C_out) {
    extern __shared__ __align__(1024) char smem_raw[];
    // Layouts: A is 128×K, B is K×128 in NVFP4.
    // Two SMEM tiles for double-buffer; A tile 0 / 1 / B tile 0 / 1.
    uint8_t* A_buf[2] = { (uint8_t*)smem_raw, (uint8_t*)smem_raw + 4096 };
    uint8_t* B_buf[2] = { (uint8_t*)smem_raw + 8192, (uint8_t*)smem_raw + 12288 };

    // A-tilt: A_data alternates dense/sparse popcount; B_data fixed all-zero
    //         -> baseline shows A>>B power
    // B-tilt: swap which side toggles
    // ...
    // (full TMA setup elided — same as v46 pattern, 8 KB tiles, mbarriers)

    // Issue tcgen05.mma in a K_DEPTH loop:
    //   for k in K_DEPTH:
    //     if MODE==3 or k_iter == 0: load_A_tile()
    //     load_B_tile()
    //     if MODE==2: TMA_multicast B to all peers
    //     tcgen05.mma.cta_group::1.kind::mxf4 [d], [a], [b], [scaleA], [scaleB], 1;
    //     fence.async tcgen05;

    // ... timing + power-probe via NVML in host loop
}

int main() {
    CK(cudaSetDevice(0));
    // Allocate A and B with two contents:
    //   "static": all-zero
    //   "toggle": random with ~16 popcount per byte (peak power per project_b300_power_data_dep)
    // ...

    printf("=== V56 NVFP4 A:B mechanism discriminator ===\n");
    printf("Each row = 1 mode × 1 contents config; record W (NVML), TFLOPS, ncu pipe util.\n");
    printf("mode  cluster  A_state  B_state   W_avg   TFLOPS  pipe_tensor_pct\n");
    // Drive with NVML sampling at 100 Hz during a 5-sec sustained run per config.
    // Collect: rows for (M0..M4) × (Astatic/Atoggle) × (Bstatic/Btoggle).
    return 0;
}
```

**Power table predictions (W per CTA at 1005 MHz, baseline B-static A-toggle = 600 W reference):**

| Mode | What it changes | Predicts which mechanism if power asymmetry inverts/equalizes |
|---|---|---|
| 0 (base) | none | reference |
| 1 (swap A/B contents) | If swap also flips A>>B → it's CONTENTS not pipeline | rules out asymmetric pipeline (#2/#3/#4) → confirms data-side |
| 2 (B multicast too) | If A=B power gap closes → multicast is the cause (#1) | confirms #1 |
| 3 (single-buffer A) | If A>>B gap GROWS → A dwell time matters (#3) | confirms #3 |
| 4 (cluster=1) | If A=B equalize → multicast was the cause (#1) | confirms #1 |

**Decision rule (decision tree):**
1. Mode 1 inverts → mechanism is purely contents-driven; mechanisms #1-4 all wrong; revisit data-dep (project_b300_power_data_dep).
2. Mode 2 equalizes AND mode 4 equalizes → **#1 multicast** is the mechanism.
3. Mode 3 amplifies AND mode 1 does NOT invert → **#3 dwell time** is mechanism.
4. None of 1-4 changes the asymmetry meaningfully (<5% W shift) → **#2 swap** (operand asymmetry built into MMA path, not data/transport).

**ncu metrics** (the harder ones):
```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active   # caveat: doesn't track tcgen05 well
lts__t_sectors_aperture_device_op_read.sum                      # multicast vs unicast traffic
sm__inst_executed_pipe_tex.sum                                  # TMA issue count
dram__bytes_read.sum.per_second                                 # if multicast, B side BW changes
```
NVML power: 100 Hz sample, 5 s sustained, mean of last 4 s.

---

## Most-impactful proposal — under 300 words

**SKETCH 1 (`v52_dual_issue_warp_sweep.cu`) is the highest-leverage retest.**

V49's 55% and V50's 74% have been propagated as "B300 dispatch is capped" and have anchored at least three downstream documents (DUAL_ISSUE_DOUBT_REPORT, V49 commit, V50 commit). If the ceiling is actually an under-occupancy artifact, then a non-trivial chunk of the catalog's "dispatch limit ≈ 128 inst/SM/cy" claim is wrong, which in turn affects the FFMA peak interpretation, the NVFP4 throughput model, and how power-vs-throughput tradeoffs are reported.

The sketch is cheap to run (one binary, five-minute sweep), one knob (`__launch_bounds__` warps × CTAs/SM), and the decision rule is binary: if `smsp__inst_issued.per_cycle_active` plateaus at ≤1.05 across 1→8 warps/SMSP the dispatch cap is real; if it climbs past 1.3 then V49 was occupancy-bound and the "55% efficient" headline collapses.

The cross-validation chain is also strong: ncu's `pipe_fma_cycles_active + pipe_alu_cycles_active` should sum to >130% if the pipes overlap, and SASS verification (single FFMA + single LOP3 per body iteration) is trivial to confirm. None of the other four sketches has a comparable downstream blast radius — sketches 2-4 refine specific numbers, sketch 5 is exploratory mechanism work — but Sketch 1 directly tests an ARCHITECTURAL claim that's already shaping how new benchmarks get designed in this repo.

Recommend running V52 first, then propagating the result back to V49/V50/DUAL_ISSUE_DOUBT_REPORT before any further pipe-cap claims are made.
