// dsmem_definitive.cu — Definitive DSMEM vs local SMEM latency and throughput test
//
// RESOLVES contradiction between:
//   B300_PIPE_CATALOG §30.H: "only 0.8% slower than local SMEM"
//   AUDIT_NOTES / dsmem_v2.cu:  "4.7× slower"
//
// ROOT CAUSE OF CONTRADICTION:
//   §30.H measured ~23 cy/iter with an XOR-accumulator loop reading the SAME
//   address each iteration. The compiler emits ONE load outside the loop
//   (LICM), making the measured cycles = loop-control overhead, not load latency.
//   dsmem_v2 used wall-clock bandwidth but with a single-thread accumulator
//   (FADD dependent chain), serializing what should be pipelined — the 4.7×
//   ratio comes from LATENCY exposure, not bandwidth limit.
//
// THIS TEST:
//   1. Latency:    Pointer-chain — each load's result feeds next address.
//                  Compiler cannot overlap; measures pure load-to-use latency.
//   2. Throughput: ILP=8 independent chains per warp — fills load pipeline.
//   Both anti-DCE: result XORed together and written to global unconditionally.
//
// Compile: nvcc -arch=sm_103a -O3 -o dsmem_definitive investigations/dsmem_definitive.cu
// Prereq:  nvidia-smi -lgc 2032 -i 0

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

namespace cg = cooperative_groups;

#define CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// ─── Configuration ────────────────────────────────────────────────────────────
static const int SMEM_WORDS  = 512;    // 2 KB shared memory — fits in cluster
static const int LAT_ITERS   = 1024;   // dependent-chain loads for latency
static const int TP_ITERS    = 512;    // iterations per independent chain
static const int ILP         = 8;      // parallel chains for throughput

// ─────────────────────────────────────────────────────────────────────────────
// Helper macro: pointer-chain load from shared memory (local)
// Reads smem[cur], writes new cur. Each call depends on previous cur.
// ─────────────────────────────────────────────────────────────────────────────
#define LOC_LOAD(cur, base_reg) \
    asm volatile("ld.shared.u32 %0, [%1];" \
                 : "=r"(cur) : "r"((base_reg) + (cur)*4u) : "memory");

// ─────────────────────────────────────────────────────────────────────────────
// Helper macro: pointer-chain load from DSMEM (cluster peer smem)
// Uses mapa to translate local smem address to peer's address space.
// ─────────────────────────────────────────────────────────────────────────────
#define DSMEM_LOAD(cur, local_base, peer_rank_val) \
    do { \
        unsigned _pa; \
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" \
                     : "=r"(_pa) \
                     : "r"((local_base) + (cur)*4u), "r"((unsigned)(peer_rank_val)) \
                     :); \
        asm volatile("ld.shared::cluster.u32 %0, [%1];" \
                     : "=r"(cur) : "r"(_pa) : "memory"); \
    } while(0)

// ─────────────────────────────────────────────────────────────────────────────
// LATENCY KERNELS — single warp (tid<32), pointer-chain
// ─────────────────────────────────────────────────────────────────────────────

// Init helper: fill smem with a pseudo-random walk table.
// Each smem[i] = some j in [0, SMEM_WORDS), so the chain won't trivially
// converge to slot 0.  Values must be runtime-derived (use seed arg).
__device__ __forceinline__
void init_smem_walk(unsigned* smem, int tid, int seed, unsigned rank_salt)
{
    if (tid < SMEM_WORDS) {
        // LCG: deterministic but not compile-time-constant (seed is a runtime arg)
        unsigned v = (unsigned)(tid) * 2654435761u ^ (unsigned)(seed) ^ rank_salt;
        // Bias toward non-trivial next slot
        smem[tid] = (v >> 5) & (SMEM_WORDS - 1);
    }
    __syncthreads();
}

// ── local SMEM latency, cluster=1 ────────────────────────────────────────────
__global__ void __cluster_dims__(1,1,1)
lat_local(unsigned long long* out, int seed)
{
    __shared__ unsigned smem[SMEM_WORDS];
    int tid = threadIdx.x;
    init_smem_walk(smem, tid, seed, 0u);

    if (tid >= 32) return;

    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur  = (unsigned)(tid * 31u + (unsigned)seed) & (SMEM_WORDS - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        LOC_LOAD(cur, base);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;   // anti-DCE
    }
}

// ── DSMEM latency, cluster=2 ─────────────────────────────────────────────────
__global__ void __cluster_dims__(2,1,1)
lat_dsmem2(unsigned long long* out, int seed)
{
    auto cluster = cg::this_cluster();
    __shared__ unsigned smem[SMEM_WORDS];

    int      tid  = threadIdx.x;
    unsigned rank = (unsigned)cluster.block_rank();
    init_smem_walk(smem, tid, seed, rank * 0xDEADBEEFu);

    cluster.sync();   // all blocks' smem ready before timing

    if (tid >= 32) return;

    unsigned peer_rank = rank ^ 1u;
    unsigned base      = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur       = (unsigned)(tid * 31u + (unsigned)seed + rank) & (SMEM_WORDS - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        DSMEM_LOAD(cur, base, peer_rank);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;
    }
}

// ── DSMEM latency, cluster=4 ─────────────────────────────────────────────────
__global__ void __cluster_dims__(4,1,1)
lat_dsmem4(unsigned long long* out, int seed)
{
    auto cluster = cg::this_cluster();
    __shared__ unsigned smem[SMEM_WORDS];

    int      tid  = threadIdx.x;
    unsigned rank = (unsigned)cluster.block_rank();
    init_smem_walk(smem, tid, seed, rank * 0xDEADBEEFu);

    cluster.sync();

    if (tid >= 32) return;

    unsigned peer_rank = (rank + 1u) & 3u;
    unsigned base      = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur       = (unsigned)(tid * 31u + (unsigned)seed + rank) & (SMEM_WORDS - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        DSMEM_LOAD(cur, base, peer_rank);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;
    }
}

// ── DSMEM latency, cluster=8 ─────────────────────────────────────────────────
__global__ void __cluster_dims__(8,1,1)
lat_dsmem8(unsigned long long* out, int seed)
{
    auto cluster = cg::this_cluster();
    __shared__ unsigned smem[SMEM_WORDS];

    int      tid  = threadIdx.x;
    unsigned rank = (unsigned)cluster.block_rank();
    init_smem_walk(smem, tid, seed, rank * 0xDEADBEEFu);

    cluster.sync();

    if (tid >= 32) return;

    unsigned peer_rank = (rank + 1u) & 7u;
    unsigned base      = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur       = (unsigned)(tid * 31u + (unsigned)seed + rank) & (SMEM_WORDS - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        DSMEM_LOAD(cur, base, peer_rank);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;
    }
}

// ── DSMEM latency, cluster=16 (non-portable) ─────────────────────────────────
__global__ void __cluster_dims__(16,1,1)
lat_dsmem16(unsigned long long* out, int seed)
{
    auto cluster = cg::this_cluster();
    __shared__ unsigned smem[SMEM_WORDS];

    int      tid  = threadIdx.x;
    unsigned rank = (unsigned)cluster.block_rank();
    init_smem_walk(smem, tid, seed, rank * 0xDEADBEEFu);

    cluster.sync();

    if (tid >= 32) return;

    unsigned peer_rank = (rank + 1u) & 15u;
    unsigned base      = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur       = (unsigned)(tid * 31u + (unsigned)seed + rank) & (SMEM_WORDS - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        DSMEM_LOAD(cur, base, peer_rank);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THROUGHPUT KERNELS — ILP=8 parallel chains, full warp (32 threads)
// 8 independent accumulators per thread; hardware can overlap their loads.
// Metric: total_loads / total_cycles = (TP_ITERS * 8) / cycles  per warp
// ─────────────────────────────────────────────────────────────────────────────

// ── local SMEM throughput ─────────────────────────────────────────────────────
__global__ void __cluster_dims__(1,1,1)
tp_local(unsigned long long* out, int seed)
{
    __shared__ unsigned smem[SMEM_WORDS];
    int tid = threadIdx.x;
    init_smem_walk(smem, tid, seed, 0u);

    if (tid >= 32) return;

    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    // 8 independent start positions
    unsigned c0 = (unsigned)(tid*7+seed+0) & (SMEM_WORDS-1);
    unsigned c1 = (unsigned)(tid*7+seed+1) & (SMEM_WORDS-1);
    unsigned c2 = (unsigned)(tid*7+seed+2) & (SMEM_WORDS-1);
    unsigned c3 = (unsigned)(tid*7+seed+3) & (SMEM_WORDS-1);
    unsigned c4 = (unsigned)(tid*7+seed+4) & (SMEM_WORDS-1);
    unsigned c5 = (unsigned)(tid*7+seed+5) & (SMEM_WORDS-1);
    unsigned c6 = (unsigned)(tid*7+seed+6) & (SMEM_WORDS-1);
    unsigned c7 = (unsigned)(tid*7+seed+7) & (SMEM_WORDS-1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        LOC_LOAD(c0, base);
        LOC_LOAD(c1, base);
        LOC_LOAD(c2, base);
        LOC_LOAD(c3, base);
        LOC_LOAD(c4, base);
        LOC_LOAD(c5, base);
        LOC_LOAD(c6, base);
        LOC_LOAD(c7, base);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ── DSMEM throughput, cluster=2 ───────────────────────────────────────────────
__global__ void __cluster_dims__(2,1,1)
tp_dsmem2(unsigned long long* out, int seed)
{
    auto cluster = cg::this_cluster();
    __shared__ unsigned smem[SMEM_WORDS];

    int      tid  = threadIdx.x;
    unsigned rank = (unsigned)cluster.block_rank();
    init_smem_walk(smem, tid, seed, rank * 0xDEADBEEFu);
    cluster.sync();

    if (tid >= 32) return;

    unsigned peer_rank = rank ^ 1u;
    unsigned base      = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned c0 = (unsigned)(tid*7+seed+0) & (SMEM_WORDS-1);
    unsigned c1 = (unsigned)(tid*7+seed+1) & (SMEM_WORDS-1);
    unsigned c2 = (unsigned)(tid*7+seed+2) & (SMEM_WORDS-1);
    unsigned c3 = (unsigned)(tid*7+seed+3) & (SMEM_WORDS-1);
    unsigned c4 = (unsigned)(tid*7+seed+4) & (SMEM_WORDS-1);
    unsigned c5 = (unsigned)(tid*7+seed+5) & (SMEM_WORDS-1);
    unsigned c6 = (unsigned)(tid*7+seed+6) & (SMEM_WORDS-1);
    unsigned c7 = (unsigned)(tid*7+seed+7) & (SMEM_WORDS-1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        DSMEM_LOAD(c0, base, peer_rank);
        DSMEM_LOAD(c1, base, peer_rank);
        DSMEM_LOAD(c2, base, peer_rank);
        DSMEM_LOAD(c3, base, peer_rank);
        DSMEM_LOAD(c4, base, peer_rank);
        DSMEM_LOAD(c5, base, peer_rank);
        DSMEM_LOAD(c6, base, peer_rank);
        DSMEM_LOAD(c7, base, peer_rank);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ── DSMEM throughput, cluster=4 ───────────────────────────────────────────────
__global__ void __cluster_dims__(4,1,1)
tp_dsmem4(unsigned long long* out, int seed)
{
    auto cluster = cg::this_cluster();
    __shared__ unsigned smem[SMEM_WORDS];

    int      tid  = threadIdx.x;
    unsigned rank = (unsigned)cluster.block_rank();
    init_smem_walk(smem, tid, seed, rank * 0xDEADBEEFu);
    cluster.sync();

    if (tid >= 32) return;

    unsigned peer_rank = (rank + 1u) & 3u;
    unsigned base      = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned c0 = (unsigned)(tid*7+seed+0) & (SMEM_WORDS-1);
    unsigned c1 = (unsigned)(tid*7+seed+1) & (SMEM_WORDS-1);
    unsigned c2 = (unsigned)(tid*7+seed+2) & (SMEM_WORDS-1);
    unsigned c3 = (unsigned)(tid*7+seed+3) & (SMEM_WORDS-1);
    unsigned c4 = (unsigned)(tid*7+seed+4) & (SMEM_WORDS-1);
    unsigned c5 = (unsigned)(tid*7+seed+5) & (SMEM_WORDS-1);
    unsigned c6 = (unsigned)(tid*7+seed+6) & (SMEM_WORDS-1);
    unsigned c7 = (unsigned)(tid*7+seed+7) & (SMEM_WORDS-1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        DSMEM_LOAD(c0, base, peer_rank);
        DSMEM_LOAD(c1, base, peer_rank);
        DSMEM_LOAD(c2, base, peer_rank);
        DSMEM_LOAD(c3, base, peer_rank);
        DSMEM_LOAD(c4, base, peer_rank);
        DSMEM_LOAD(c5, base, peer_rank);
        DSMEM_LOAD(c6, base, peer_rank);
        DSMEM_LOAD(c7, base, peer_rank);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ── DSMEM throughput, cluster=8 ───────────────────────────────────────────────
__global__ void __cluster_dims__(8,1,1)
tp_dsmem8(unsigned long long* out, int seed)
{
    auto cluster = cg::this_cluster();
    __shared__ unsigned smem[SMEM_WORDS];

    int      tid  = threadIdx.x;
    unsigned rank = (unsigned)cluster.block_rank();
    init_smem_walk(smem, tid, seed, rank * 0xDEADBEEFu);
    cluster.sync();

    if (tid >= 32) return;

    unsigned peer_rank = (rank + 1u) & 7u;
    unsigned base      = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned c0 = (unsigned)(tid*7+seed+0) & (SMEM_WORDS-1);
    unsigned c1 = (unsigned)(tid*7+seed+1) & (SMEM_WORDS-1);
    unsigned c2 = (unsigned)(tid*7+seed+2) & (SMEM_WORDS-1);
    unsigned c3 = (unsigned)(tid*7+seed+3) & (SMEM_WORDS-1);
    unsigned c4 = (unsigned)(tid*7+seed+4) & (SMEM_WORDS-1);
    unsigned c5 = (unsigned)(tid*7+seed+5) & (SMEM_WORDS-1);
    unsigned c6 = (unsigned)(tid*7+seed+6) & (SMEM_WORDS-1);
    unsigned c7 = (unsigned)(tid*7+seed+7) & (SMEM_WORDS-1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        DSMEM_LOAD(c0, base, peer_rank);
        DSMEM_LOAD(c1, base, peer_rank);
        DSMEM_LOAD(c2, base, peer_rank);
        DSMEM_LOAD(c3, base, peer_rank);
        DSMEM_LOAD(c4, base, peer_rank);
        DSMEM_LOAD(c5, base, peer_rank);
        DSMEM_LOAD(c6, base, peer_rank);
        DSMEM_LOAD(c7, base, peer_rank);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ── DSMEM throughput, cluster=16 ──────────────────────────────────────────────
__global__ void __cluster_dims__(16,1,1)
tp_dsmem16(unsigned long long* out, int seed)
{
    auto cluster = cg::this_cluster();
    __shared__ unsigned smem[SMEM_WORDS];

    int      tid  = threadIdx.x;
    unsigned rank = (unsigned)cluster.block_rank();
    init_smem_walk(smem, tid, seed, rank * 0xDEADBEEFu);
    cluster.sync();

    if (tid >= 32) return;

    unsigned peer_rank = (rank + 1u) & 15u;
    unsigned base      = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned c0 = (unsigned)(tid*7+seed+0) & (SMEM_WORDS-1);
    unsigned c1 = (unsigned)(tid*7+seed+1) & (SMEM_WORDS-1);
    unsigned c2 = (unsigned)(tid*7+seed+2) & (SMEM_WORDS-1);
    unsigned c3 = (unsigned)(tid*7+seed+3) & (SMEM_WORDS-1);
    unsigned c4 = (unsigned)(tid*7+seed+4) & (SMEM_WORDS-1);
    unsigned c5 = (unsigned)(tid*7+seed+5) & (SMEM_WORDS-1);
    unsigned c6 = (unsigned)(tid*7+seed+6) & (SMEM_WORDS-1);
    unsigned c7 = (unsigned)(tid*7+seed+7) & (SMEM_WORDS-1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        DSMEM_LOAD(c0, base, peer_rank);
        DSMEM_LOAD(c1, base, peer_rank);
        DSMEM_LOAD(c2, base, peer_rank);
        DSMEM_LOAD(c3, base, peer_rank);
        DSMEM_LOAD(c4, base, peer_rank);
        DSMEM_LOAD(c5, base, peer_rank);
        DSMEM_LOAD(c6, base, peer_rank);
        DSMEM_LOAD(c7, base, peer_rank);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host: run kernel, collect cycles from block 0, compute stats
// ─────────────────────────────────────────────────────────────────────────────

static double run_lat(void* fn, int cluster_size, int nblocks,
                      unsigned long long* d_out, int seed, bool non_portable)
{
    // Set non-portable cluster attribute if needed
    if (non_portable) {
        CHECK(cudaFuncSetAttribute(fn,
              cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    }

    int smem_bytes = SMEM_WORDS * 4;
    CHECK(cudaFuncSetAttribute(fn,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = cluster_size;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;

    cudaLaunchConfig_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.gridDim      = dim3(nblocks);
    cfg.blockDim     = dim3(256);  // only warp 0 active; extra threads exit early
    cfg.dynamicSmemBytes = 0;      // smem declared __shared__, not dynamic
    cfg.stream       = nullptr;
    cfg.attrs        = &attr;
    cfg.numAttrs     = 1;

    void* args[] = { &d_out, &seed };

    // Warmup
    for (int i = 0; i < 3; i++) {
        CHECK(cudaLaunchKernelExC(&cfg, fn, args));
    }
    CHECK(cudaDeviceSynchronize());

    // Timed runs — take median of 7 runs from block 0
    static unsigned long long h_buf[256];
    double best_cy = 1e18;
    for (int r = 0; r < 10; r++) {
        CHECK(cudaLaunchKernelExC(&cfg, fn, args));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_buf, d_out, nblocks * 2 * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
        double cy = (double)h_buf[0];   // block 0, slot 0 = cycles
        if (cy < best_cy) best_cy = cy;
    }
    return best_cy;   // raw cycle count for LAT_ITERS loads
}

static double run_tp(void* fn, int cluster_size, int nblocks,
                     unsigned long long* d_out, int seed, bool non_portable)
{
    if (non_portable) {
        CHECK(cudaFuncSetAttribute(fn,
              cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    }

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = cluster_size;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;

    cudaLaunchConfig_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.gridDim      = dim3(nblocks);
    cfg.blockDim     = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream       = nullptr;
    cfg.attrs        = &attr;
    cfg.numAttrs     = 1;

    void* args[] = { &d_out, &seed };

    for (int i = 0; i < 3; i++) {
        CHECK(cudaLaunchKernelExC(&cfg, fn, args));
    }
    CHECK(cudaDeviceSynchronize());

    static unsigned long long h_buf[256];
    double best_cy = 1e18;
    for (int r = 0; r < 10; r++) {
        CHECK(cudaLaunchKernelExC(&cfg, fn, args));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_buf, d_out, nblocks * 2 * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
        double cy = (double)h_buf[0];
        if (cy < best_cy) best_cy = cy;
    }
    return best_cy;
}

int main()
{
    cudaDeviceProp prop;
    CHECK(cudaSetDevice(0));
    CHECK(cudaGetDeviceProperties(&prop, 0));

    int sm_count = prop.multiProcessorCount;
    printf("# GPU: %s, SMs: %d\n",
           prop.name, sm_count);

    unsigned long long* d_out;
    // We need 2 ull per block; max blocks = sm_count (cluster=1 case)
    CHECK(cudaMalloc(&d_out, sm_count * 2 * sizeof(unsigned long long)));

    int seed = 0x1234ABCD;

    // ── LATENCY TABLE ──────────────────────────────────────────────────────────
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("LATENCY: dependent pointer-chain (%d loads), single warp\n", LAT_ITERS);
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("%-25s %8s %10s %10s\n",
           "variant", "cy/load", "raw_cy", "vs_local");

    auto lat_report = [&](const char* name, double raw_cy, double local_cy) {
        double cy_per_load = raw_cy / LAT_ITERS;
        double ratio       = raw_cy / local_cy;
        printf("%-25s %8.2f %10.0f %10.2fx\n",
               name, cy_per_load, raw_cy, ratio);
    };

    // local SMEM (cluster=1, 1 block)
    double loc_lat_cy = run_lat((void*)lat_local, 1, 1, d_out, seed, false);
    lat_report("local SMEM (cluster=1)", loc_lat_cy, loc_lat_cy);

    // DSMEM cluster=2 (use 2 blocks)
    double d2_lat_cy = run_lat((void*)lat_dsmem2, 2, 2, d_out, seed, false);
    lat_report("DSMEM cluster=2", d2_lat_cy, loc_lat_cy);

    // DSMEM cluster=4
    double d4_lat_cy = run_lat((void*)lat_dsmem4, 4, 4, d_out, seed, false);
    lat_report("DSMEM cluster=4", d4_lat_cy, loc_lat_cy);

    // DSMEM cluster=8
    double d8_lat_cy = run_lat((void*)lat_dsmem8, 8, 8, d_out, seed, false);
    lat_report("DSMEM cluster=8", d8_lat_cy, loc_lat_cy);

    // DSMEM cluster=16 (non-portable sm_103a)
    double d16_lat_cy = run_lat((void*)lat_dsmem16, 16, 16, d_out, seed, true);
    lat_report("DSMEM cluster=16 (non-port)", d16_lat_cy, loc_lat_cy);

    // ── THROUGHPUT TABLE ───────────────────────────────────────────────────────
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("THROUGHPUT: ILP=%d chains, %d loads/chain, single warp\n", ILP, TP_ITERS);
    printf("Metric: total_loads_issued / elapsed_cycles  (higher = better)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("%-25s %10s %10s %12s %10s\n",
           "variant", "cy/8loads", "loads/cy", "raw_cy", "vs_local");

    auto tp_report = [&](const char* name, double raw_cy, double local_cy) {
        // TP_ITERS iterations × ILP loads each = total loads
        double total_loads = (double)TP_ITERS * ILP;
        double loads_per_cy = total_loads / raw_cy;
        double cy_per_8     = raw_cy / TP_ITERS;  // cycles for one group of 8 loads
        double ratio        = raw_cy / local_cy;
        printf("%-25s %10.2f %10.3f %12.0f %10.2fx\n",
               name, cy_per_8, loads_per_cy, raw_cy, ratio);
    };

    double loc_tp_cy = run_tp((void*)tp_local,   1,  1, d_out, seed, false);
    tp_report("local SMEM (cluster=1)", loc_tp_cy, loc_tp_cy);

    double d2_tp_cy  = run_tp((void*)tp_dsmem2,  2,  2, d_out, seed, false);
    tp_report("DSMEM cluster=2", d2_tp_cy, loc_tp_cy);

    double d4_tp_cy  = run_tp((void*)tp_dsmem4,  4,  4, d_out, seed, false);
    tp_report("DSMEM cluster=4", d4_tp_cy, loc_tp_cy);

    double d8_tp_cy  = run_tp((void*)tp_dsmem8,  8,  8, d_out, seed, false);
    tp_report("DSMEM cluster=8", d8_tp_cy, loc_tp_cy);

    double d16_tp_cy = run_tp((void*)tp_dsmem16, 16, 16, d_out, seed, true);
    tp_report("DSMEM cluster=16 (non-port)", d16_tp_cy, loc_tp_cy);

    // ── SUMMARY ────────────────────────────────────────────────────────────────
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Local SMEM latency:        %.2f cy/load\n",
           loc_lat_cy / LAT_ITERS);
    printf("DSMEM latency (cluster=2): %.2f cy/load  (%.2fx local)\n",
           d2_lat_cy / LAT_ITERS, d2_lat_cy / loc_lat_cy);
    printf("DSMEM latency (cluster=4): %.2f cy/load  (%.2fx local)\n",
           d4_lat_cy / LAT_ITERS, d4_lat_cy / loc_lat_cy);
    printf("\n");
    printf("Local SMEM throughput (ILP=8):   %.3f loads/cy\n",
           (double)TP_ITERS * ILP / loc_tp_cy);
    printf("DSMEM throughput (cluster=2 ILP=8): %.3f loads/cy  (%.2fx local)\n",
           (double)TP_ITERS * ILP / d2_tp_cy, d2_tp_cy / loc_tp_cy);
    printf("DSMEM throughput (cluster=4 ILP=8): %.3f loads/cy  (%.2fx local)\n",
           (double)TP_ITERS * ILP / d4_tp_cy, d4_tp_cy / loc_tp_cy);
    printf("\n");
    printf("Clock @2032 MHz: 1 cy = %.3f ns\n", 1000.0/2032.0);

    CHECK(cudaFree(d_out));
    return 0;
}
