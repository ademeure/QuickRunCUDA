// dsmem_definitive.cu — Definitive DSMEM vs local SMEM latency and throughput test
//
// RESOLVES contradiction between:
//   B300_PIPE_CATALOG §30.H: "only 0.8% slower than local SMEM"
//   AUDIT_NOTES / dsmem_v2.cu:  "4.7× slower"
//
// ROOT CAUSE OF CONTRADICTION:
//   §30.H: The XOR-accumulator loop read from a FIXED address each iteration
//   (remote_addr + threadIdx.x*4 — constant per thread).  The compiler could
//   perform LICM: load once before the loop, then XOR in a tight constant loop.
//   The measured ~23 cy/iter is the XOR + loop overhead, NOT the load latency.
//   dsmem_v2: used wall-clock / bandwidth approach with a single-thread FADD
//   accumulator, which serializes loads through an FP dependency chain, measuring
//   latency not throughput.  But more critically, dsmem_v2.cu compiles its SMEM
//   init at compile time (rank*1000+i is partially constant), enabling DCE.
//
// THIS TEST: constructs a dependent pointer chain where EACH load's result
//   feeds the ADDRESS of the next load.  The compiler cannot hoist, reorder,
//   or eliminate any load.  Verified in SASS: LDS R0,[R0] for local SMEM.
//   DSMEM uses the exact PTX pattern from bench_dsmem.cu (which was validated
//   correct in §30.H validation table) but with the dependent-chain structure.
//
// Compile: nvcc -arch=sm_103a -O3 -o dsmem_definitive investigations/dsmem_definitive.cu
// Prereq:  nvidia-smi -lgc 2032 -i 0

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// ─── Configuration ────────────────────────────────────────────────────────────
static const int SMEM_WORDS  = 512;    // 2 KB smem (512 × u32)
static const int LAT_ITERS   = 1024;   // serial loads for latency
static const int TP_ITERS    = 512;    // iterations per independent chain
static const int ILP         = 8;      // parallel independent chains

// ─────────────────────────────────────────────────────────────────────────────
// LATENCY KERNELS — single warp (tid<32), dependent pointer chain
// LAT: each load's result is the index for the next load.
//   smem[i] = runtime-computed value in [0, SMEM_WORDS)
//   cur = smem[cur & (SMEM_WORDS-1)]  →  address depends on previous value
// DSMEM kernels: cluster.sync() BEFORE the clock start.
// ─────────────────────────────────────────────────────────────────────────────

// ── LOCAL SMEM latency (cluster=1) ───────────────────────────────────────────
extern __shared__ unsigned smem_dyn[];

__global__ void __cluster_dims__(1,1,1)
lat_local(unsigned long long* out, int seed)
{
    // Use dynamic shared memory so init happens at runtime (defeats DCE)
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    // Init: runtime-derived next-pointers
    // Value at slot i = (i * LCG ^ seed) & mask — not a compile-time constant
    if (tid < SMEM_WORDS) {
        unsigned v = (unsigned)tid * 2654435761u ^ (unsigned)seed;
        smem[tid]  = (v >> 5) & (SMEM_WORDS - 1u);
    }
    __syncthreads();

    if (tid >= 32) return;  // only warp 0 measures

    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur = (unsigned)(tid * 31 + seed) & (SMEM_WORDS - 1u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    // Dependent chain: each LDS address comes from previous LDS result
    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(smem_addr + cur * 4u)
                     : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;   // anti-DCE: nonzero proves loop ran
    }
}

// ── DSMEM latency (cluster=2) ─────────────────────────────────────────────────
// Uses the EXACT PTX pattern from bench_dsmem.cu (validated correct):
//   mapa.shared::cluster.u32 remote_addr, local_addr, target_cta
//   ld.shared::cluster.u32  result, [remote_addr]
// But with dependent chain: remote_addr is recomputed each iteration from result.
__global__ void __cluster_dims__(2,1,1)
lat_dsmem2(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta_rank));

    // Init local smem with runtime values
    if (tid < SMEM_WORDS) {
        unsigned v = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_cta_rank * 0xDEADBEEFu);
        smem[tid]  = (v >> 5) & (SMEM_WORDS - 1u);
    }
    __syncthreads();

    // Cluster barrier: all blocks' smem populated before timing
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned target_cta = my_cta_rank ^ 1u;  // neighbor in cluster=2
    unsigned smem_base  = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur = (unsigned)(tid * 31 + seed + my_cta_rank) & (SMEM_WORDS - 1u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    // Dependent DSMEM chain
    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        unsigned remote_addr;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(remote_addr)
                     : "r"(smem_base + cur * 4u), "r"(target_cta));
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(remote_addr)
                     : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;
    }
}

// ── DSMEM latency (cluster=4) ─────────────────────────────────────────────────
__global__ void __cluster_dims__(4,1,1)
lat_dsmem4(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta_rank));

    if (tid < SMEM_WORDS) {
        unsigned v = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_cta_rank * 0xDEADBEEFu);
        smem[tid]  = (v >> 5) & (SMEM_WORDS - 1u);
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned target_cta = (my_cta_rank + 1u) & 3u;
    unsigned smem_base  = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur = (unsigned)(tid * 31 + seed + my_cta_rank) & (SMEM_WORDS - 1u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        unsigned remote_addr;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(remote_addr)
                     : "r"(smem_base + cur * 4u), "r"(target_cta));
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(remote_addr)
                     : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;
    }
}

// ── DSMEM latency (cluster=8) ─────────────────────────────────────────────────
__global__ void __cluster_dims__(8,1,1)
lat_dsmem8(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta_rank));

    if (tid < SMEM_WORDS) {
        unsigned v = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_cta_rank * 0xDEADBEEFu);
        smem[tid]  = (v >> 5) & (SMEM_WORDS - 1u);
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned target_cta = (my_cta_rank + 1u) & 7u;
    unsigned smem_base  = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur = (unsigned)(tid * 31 + seed + my_cta_rank) & (SMEM_WORDS - 1u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        unsigned remote_addr;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(remote_addr)
                     : "r"(smem_base + cur * 4u), "r"(target_cta));
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(remote_addr)
                     : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;
    }
}

// ── DSMEM latency (cluster=16, non-portable) ──────────────────────────────────
__global__ void __cluster_dims__(16,1,1)
lat_dsmem16(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta_rank));

    if (tid < SMEM_WORDS) {
        unsigned v = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_cta_rank * 0xDEADBEEFu);
        smem[tid]  = (v >> 5) & (SMEM_WORDS - 1u);
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned target_cta = (my_cta_rank + 1u) & 15u;
    unsigned smem_base  = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur = (unsigned)(tid * 31 + seed + my_cta_rank) & (SMEM_WORDS - 1u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        unsigned remote_addr;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(remote_addr)
                     : "r"(smem_base + cur * 4u), "r"(target_cta));
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(remote_addr)
                     : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THROUGHPUT KERNELS — ILP=8 parallel chains, single warp (tid<32)
// 8 independent accumulators; hardware can pipeline all 8 loads per iteration.
// ─────────────────────────────────────────────────────────────────────────────

// ── LOCAL SMEM throughput ─────────────────────────────────────────────────────
__global__ void __cluster_dims__(1,1,1)
tp_local(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    if (tid < SMEM_WORDS) {
        unsigned v = (unsigned)tid * 2654435761u ^ (unsigned)seed;
        smem[tid]  = (v >> 5) & (SMEM_WORDS - 1u);
    }
    __syncthreads();

    if (tid >= 32) return;

    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    // 8 independent start positions
    unsigned c0 = (unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u);
    unsigned c1 = (unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u);
    unsigned c2 = (unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u);
    unsigned c3 = (unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u);
    unsigned c4 = (unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u);
    unsigned c5 = (unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u);
    unsigned c6 = (unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u);
    unsigned c7 = (unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c0) : "r"(base+c0*4u) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c1) : "r"(base+c1*4u) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c2) : "r"(base+c2*4u) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c3) : "r"(base+c3*4u) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c4) : "r"(base+c4*4u) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c5) : "r"(base+c5*4u) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c6) : "r"(base+c6*4u) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c7) : "r"(base+c7*4u) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ── DSMEM throughput (cluster=2) ──────────────────────────────────────────────
__global__ void __cluster_dims__(2,1,1)
tp_dsmem2(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta_rank));

    if (tid < SMEM_WORDS) {
        unsigned v = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_cta_rank * 0xDEADBEEFu);
        smem[tid]  = (v >> 5) & (SMEM_WORDS - 1u);
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned target_cta = my_cta_rank ^ 1u;
    unsigned base       = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned c0 = (unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u);
    unsigned c1 = (unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u);
    unsigned c2 = (unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u);
    unsigned c3 = (unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u);
    unsigned c4 = (unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u);
    unsigned c5 = (unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u);
    unsigned c6 = (unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u);
    unsigned c7 = (unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        unsigned pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa0) : "r"(base+c0*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa1) : "r"(base+c1*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa2) : "r"(base+c2*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa3) : "r"(base+c3*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa4) : "r"(base+c4*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa5) : "r"(base+c5*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa6) : "r"(base+c6*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa7) : "r"(base+c7*4u), "r"(target_cta));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c0) : "r"(pa0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c1) : "r"(pa1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c2) : "r"(pa2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c3) : "r"(pa3) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c4) : "r"(pa4) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c5) : "r"(pa5) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c6) : "r"(pa6) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c7) : "r"(pa7) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ── DSMEM throughput (cluster=4) ──────────────────────────────────────────────
__global__ void __cluster_dims__(4,1,1)
tp_dsmem4(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta_rank));

    if (tid < SMEM_WORDS) {
        unsigned v = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_cta_rank * 0xDEADBEEFu);
        smem[tid]  = (v >> 5) & (SMEM_WORDS - 1u);
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned target_cta = (my_cta_rank + 1u) & 3u;
    unsigned base       = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned c0 = (unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u);
    unsigned c1 = (unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u);
    unsigned c2 = (unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u);
    unsigned c3 = (unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u);
    unsigned c4 = (unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u);
    unsigned c5 = (unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u);
    unsigned c6 = (unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u);
    unsigned c7 = (unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        unsigned pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa0) : "r"(base+c0*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa1) : "r"(base+c1*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa2) : "r"(base+c2*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa3) : "r"(base+c3*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa4) : "r"(base+c4*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa5) : "r"(base+c5*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa6) : "r"(base+c6*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa7) : "r"(base+c7*4u), "r"(target_cta));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c0) : "r"(pa0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c1) : "r"(pa1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c2) : "r"(pa2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c3) : "r"(pa3) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c4) : "r"(pa4) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c5) : "r"(pa5) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c6) : "r"(pa6) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c7) : "r"(pa7) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ── DSMEM throughput (cluster=8) ──────────────────────────────────────────────
__global__ void __cluster_dims__(8,1,1)
tp_dsmem8(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta_rank));

    if (tid < SMEM_WORDS) {
        unsigned v = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_cta_rank * 0xDEADBEEFu);
        smem[tid]  = (v >> 5) & (SMEM_WORDS - 1u);
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned target_cta = (my_cta_rank + 1u) & 7u;
    unsigned base       = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned c0 = (unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u);
    unsigned c1 = (unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u);
    unsigned c2 = (unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u);
    unsigned c3 = (unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u);
    unsigned c4 = (unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u);
    unsigned c5 = (unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u);
    unsigned c6 = (unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u);
    unsigned c7 = (unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        unsigned pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa0) : "r"(base+c0*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa1) : "r"(base+c1*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa2) : "r"(base+c2*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa3) : "r"(base+c3*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa4) : "r"(base+c4*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa5) : "r"(base+c5*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa6) : "r"(base+c6*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa7) : "r"(base+c7*4u), "r"(target_cta));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c0) : "r"(pa0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c1) : "r"(pa1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c2) : "r"(pa2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c3) : "r"(pa3) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c4) : "r"(pa4) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c5) : "r"(pa5) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c6) : "r"(pa6) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c7) : "r"(pa7) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ── DSMEM throughput (cluster=16) ─────────────────────────────────────────────
__global__ void __cluster_dims__(16,1,1)
tp_dsmem16(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta_rank));

    if (tid < SMEM_WORDS) {
        unsigned v = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_cta_rank * 0xDEADBEEFu);
        smem[tid]  = (v >> 5) & (SMEM_WORDS - 1u);
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned target_cta = (my_cta_rank + 1u) & 15u;
    unsigned base       = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned c0 = (unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u);
    unsigned c1 = (unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u);
    unsigned c2 = (unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u);
    unsigned c3 = (unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u);
    unsigned c4 = (unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u);
    unsigned c5 = (unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u);
    unsigned c6 = (unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u);
    unsigned c7 = (unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        unsigned pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa0) : "r"(base+c0*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa1) : "r"(base+c1*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa2) : "r"(base+c2*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa3) : "r"(base+c3*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa4) : "r"(base+c4*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa5) : "r"(base+c5*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa6) : "r"(base+c6*4u), "r"(target_cta));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(pa7) : "r"(base+c7*4u), "r"(target_cta));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c0) : "r"(pa0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c1) : "r"(pa1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c2) : "r"(pa2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c3) : "r"(pa3) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c4) : "r"(pa4) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c5) : "r"(pa5) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c6) : "r"(pa6) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c7) : "r"(pa7) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host runner
// ─────────────────────────────────────────────────────────────────────────────

struct RunResult {
    double median_cy;
    unsigned long long anti_dce;
};

static RunResult run_kernel(void* fn, int cluster_size, int nblocks,
                            unsigned long long* d_out, int seed,
                            bool non_portable)
{
    int smem_bytes = SMEM_WORDS * sizeof(unsigned);  // 2 KB

    CHECK(cudaFuncSetAttribute(fn,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

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
    cfg.gridDim          = dim3(nblocks);
    cfg.blockDim         = dim3(256);  // 256 threads; only warp 0 (tid<32) measures
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream           = nullptr;
    cfg.attrs            = &attr;
    cfg.numAttrs         = 1;

    void* args[] = { &d_out, &seed };

    // Warmup
    for (int i = 0; i < 3; i++) {
        CHECK(cudaLaunchKernelExC(&cfg, fn, args));
    }
    CHECK(cudaDeviceSynchronize());

    const int NTRIALS = 12;
    static unsigned long long h_buf[512];
    double trials[NTRIALS];
    for (int r = 0; r < NTRIALS; r++) {
        CHECK(cudaLaunchKernelExC(&cfg, fn, args));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_buf, d_out,
                         (size_t)nblocks * 2 * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
        trials[r] = (double)h_buf[0];  // block 0 = first cluster's rank-0
    }

    // Sort and take middle median
    for (int i = 0; i < NTRIALS-1; i++)
        for (int j = i+1; j < NTRIALS; j++)
            if (trials[j] < trials[i]) { double t=trials[i]; trials[i]=trials[j]; trials[j]=t; }

    RunResult r;
    r.median_cy = (trials[NTRIALS/2 - 1] + trials[NTRIALS/2]) * 0.5;
    r.anti_dce  = h_buf[1];  // nonzero = DCE did not happen
    return r;
}

int main()
{
    cudaDeviceProp prop;
    CHECK(cudaSetDevice(0));
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

    int cur_clock = 0;
    {
        FILE* f = popen("nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader -i 0 2>/dev/null", "r");
        if (f) { (void)fscanf(f, "%d", &cur_clock); pclose(f); }
    }

    printf("# GPU: %s  SMs: %d  current_SM_clock: %d MHz\n",
           prop.name, sm_count, cur_clock);
    printf("# SMEM_WORDS=%d  LAT_ITERS=%d  TP_ITERS=%d  ILP=%d\n",
           SMEM_WORDS, LAT_ITERS, TP_ITERS, ILP);
    if (cur_clock < 2000)
        printf("# WARNING: clock below 2032 MHz — run: nvidia-smi -lgc 2032 -i 0\n");

    unsigned long long* d_out;
    CHECK(cudaMalloc(&d_out, (size_t)sm_count * 2 * sizeof(unsigned long long)));

    int seed = 0x1234ABCD;

    // ── LATENCY ────────────────────────────────────────────────────────────────
    printf("\n");
    printf("=================================================================\n");
    printf("LATENCY: dependent pointer chain (%d serial loads), warp 0 only\n", LAT_ITERS);
    printf("=================================================================\n");
    printf("%-30s %8s %10s %10s %8s\n",
           "variant", "cy/load", "total_cy", "vs_local", "anti_dce");

    RunResult loc_lat = run_kernel((void*)lat_local,   1,  1, d_out, seed, false);
    printf("%-30s %8.2f %10.0f %10.2fx %8llx\n",
           "local SMEM (cluster=1)",
           loc_lat.median_cy / LAT_ITERS, loc_lat.median_cy, 1.0, loc_lat.anti_dce);

    RunResult d2_lat  = run_kernel((void*)lat_dsmem2,  2,  2, d_out, seed, false);
    printf("%-30s %8.2f %10.0f %10.2fx %8llx\n",
           "DSMEM cluster=2",
           d2_lat.median_cy / LAT_ITERS, d2_lat.median_cy,
           d2_lat.median_cy / loc_lat.median_cy, d2_lat.anti_dce);

    RunResult d4_lat  = run_kernel((void*)lat_dsmem4,  4,  4, d_out, seed, false);
    printf("%-30s %8.2f %10.0f %10.2fx %8llx\n",
           "DSMEM cluster=4",
           d4_lat.median_cy / LAT_ITERS, d4_lat.median_cy,
           d4_lat.median_cy / loc_lat.median_cy, d4_lat.anti_dce);

    RunResult d8_lat  = run_kernel((void*)lat_dsmem8,  8,  8, d_out, seed, false);
    printf("%-30s %8.2f %10.0f %10.2fx %8llx\n",
           "DSMEM cluster=8",
           d8_lat.median_cy / LAT_ITERS, d8_lat.median_cy,
           d8_lat.median_cy / loc_lat.median_cy, d8_lat.anti_dce);

    RunResult d16_lat = run_kernel((void*)lat_dsmem16, 16, 16, d_out, seed, true);
    printf("%-30s %8.2f %10.0f %10.2fx %8llx\n",
           "DSMEM cluster=16 (non-port)",
           d16_lat.median_cy / LAT_ITERS, d16_lat.median_cy,
           d16_lat.median_cy / loc_lat.median_cy, d16_lat.anti_dce);

    // ── THROUGHPUT ─────────────────────────────────────────────────────────────
    printf("\n");
    printf("=================================================================\n");
    printf("THROUGHPUT: ILP=%d parallel chains × %d iters, warp 0 only\n", ILP, TP_ITERS);
    printf("ld/cy = (%d × %d) / total_cycles\n", ILP, TP_ITERS);
    printf("=================================================================\n");
    printf("%-30s %10s %8s %12s %10s\n",
           "variant", "cy/8loads", "ld/cy", "total_cy", "vs_local");

    auto tp_print = [&](const char* name, RunResult r, double local_cy) {
        double total_loads  = (double)TP_ITERS * ILP;
        double loads_per_cy = total_loads / r.median_cy;
        double cy_per_8     = r.median_cy / TP_ITERS;
        double ratio        = r.median_cy / local_cy;
        printf("%-30s %10.2f %8.3f %12.0f %10.2fx\n",
               name, cy_per_8, loads_per_cy, r.median_cy, ratio);
    };

    RunResult loc_tp = run_kernel((void*)tp_local,   1,  1, d_out, seed, false);
    tp_print("local SMEM (cluster=1)", loc_tp, loc_tp.median_cy);

    RunResult d2_tp  = run_kernel((void*)tp_dsmem2,  2,  2, d_out, seed, false);
    tp_print("DSMEM cluster=2", d2_tp, loc_tp.median_cy);

    RunResult d4_tp  = run_kernel((void*)tp_dsmem4,  4,  4, d_out, seed, false);
    tp_print("DSMEM cluster=4", d4_tp, loc_tp.median_cy);

    RunResult d8_tp  = run_kernel((void*)tp_dsmem8,  8,  8, d_out, seed, false);
    tp_print("DSMEM cluster=8", d8_tp, loc_tp.median_cy);

    RunResult d16_tp = run_kernel((void*)tp_dsmem16, 16, 16, d_out, seed, true);
    tp_print("DSMEM cluster=16 (non-port)", d16_tp, loc_tp.median_cy);

    // ── SUMMARY ────────────────────────────────────────────────────────────────
    const double MHZ = 2032.0;
    printf("\n");
    printf("=================================================================\n");
    printf("SUMMARY  (clock target: %.0f MHz = %.3f ns/cy)\n", MHZ, 1000.0/MHZ);
    printf("=================================================================\n");
    printf("Local SMEM latency:          %6.2f cy  = %.2f ns/load\n",
           loc_lat.median_cy/LAT_ITERS, (loc_lat.median_cy/LAT_ITERS)*1000.0/MHZ);
    printf("DSMEM lat (cluster=2):       %6.2f cy  = %.2f ns/load  (%+.1f cy overhead)\n",
           d2_lat.median_cy/LAT_ITERS,
           (d2_lat.median_cy/LAT_ITERS)*1000.0/MHZ,
           (d2_lat.median_cy - loc_lat.median_cy)/LAT_ITERS);
    printf("DSMEM lat (cluster=4):       %6.2f cy  = %.2f ns/load  (%+.1f cy overhead)\n",
           d4_lat.median_cy/LAT_ITERS,
           (d4_lat.median_cy/LAT_ITERS)*1000.0/MHZ,
           (d4_lat.median_cy - loc_lat.median_cy)/LAT_ITERS);
    printf("DSMEM lat (cluster=8):       %6.2f cy  = %.2f ns/load  (%+.1f cy overhead)\n",
           d8_lat.median_cy/LAT_ITERS,
           (d8_lat.median_cy/LAT_ITERS)*1000.0/MHZ,
           (d8_lat.median_cy - loc_lat.median_cy)/LAT_ITERS);
    printf("DSMEM lat (cluster=16):      %6.2f cy  = %.2f ns/load  (%+.1f cy overhead)\n",
           d16_lat.median_cy/LAT_ITERS,
           (d16_lat.median_cy/LAT_ITERS)*1000.0/MHZ,
           (d16_lat.median_cy - loc_lat.median_cy)/LAT_ITERS);
    printf("\n");
    printf("Local SMEM tp (ILP=8):            %.3f ld/cy\n",
           (double)TP_ITERS*ILP/loc_tp.median_cy);
    printf("DSMEM tp (cluster=2, ILP=8):      %.3f ld/cy  (%.2fx vs local)\n",
           (double)TP_ITERS*ILP/d2_tp.median_cy, d2_tp.median_cy/loc_tp.median_cy);
    printf("DSMEM tp (cluster=4, ILP=8):      %.3f ld/cy  (%.2fx vs local)\n",
           (double)TP_ITERS*ILP/d4_tp.median_cy, d4_tp.median_cy/loc_tp.median_cy);
    printf("DSMEM tp (cluster=8, ILP=8):      %.3f ld/cy  (%.2fx vs local)\n",
           (double)TP_ITERS*ILP/d8_tp.median_cy, d8_tp.median_cy/loc_tp.median_cy);

    CHECK(cudaFree(d_out));
    return 0;
}
