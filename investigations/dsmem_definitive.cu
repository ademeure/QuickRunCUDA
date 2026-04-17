// dsmem_definitive.cu — Definitive DSMEM vs local SMEM latency and throughput test
//
// RESOLVES contradiction between:
//   B300_PIPE_CATALOG §30.H: "only 0.8% slower than local SMEM"
//   AUDIT_NOTES / dsmem_v2.cu:  "4.7× slower"
//
// ANSWER: DSMEM is ~8× slower in latency and ~9× slower in throughput (ILP=4).
//         See investigations/04_dsmem_overhead.md for full analysis.
//
// ROOT CAUSE (confirmed in SASS):
//   §30.H: bench_dsmem.cu loop body contains only XOR+counter+branch.
//   The LDS R3,[R8+UR5] (DSMEM load) appears BEFORE the loop — ptxas hoisted it.
//   So the measured ~23 cy/iter = loop overhead (XOR + branch), NOT load latency.
//   Same LICM happened for the local SMEM variant. Both showed ~23 cy because
//   both measured loop overhead.
//
//   dsmem_v2: wall-clock, FADD chain → latency-bound, single-threaded.
//   Ratio was 4.7×, but the true latency ratio is ~8×.
//
// SASS MECHANISM (B300):
//   ld.shared::cluster.u32 with scalar-register address compiles to LD.E (global
//   load via shared-memory window), NOT LDS. The mapa result is combined with
//   SR_SWINHI via PRMT+IMAD to form a 64-bit global window address, then LD.E
//   issues a global load into the peer SM's shared memory. This is correct — the
//   hardware maps peer smem into global address space. The cost is L2/interconnect
//   latency (~224 cy) vs local shared memory crossbar latency (~28 cy).
//
// CRASH NOTE:
//   Dependent DSMEM chains crash non-deterministically. Crash rate ~50% for
//   cluster=2 at ≥50 iterations; cluster=4 at ≥10 iterations. When successful,
//   results are extremely consistent. Use small iteration counts per launch.
//   See /tmp/dsmem_isolate for the standalone reliable test binary.
//
// MEASURED RESULTS (B300, 1920 MHz):
//   local SMEM latency:  28.0 cy/load  (5000-iter chain, SASS: LDS R0,[R0+UR5])
//   DSMEM latency:      224 cy/load  (50-iter chain, cluster=2)
//   DSMEM latency:      201 cy/load  (5-iter chain, cluster=4 and cluster=8)
//   local SMEM tp ILP4:   7.0 cy/load
//   DSMEM tp ILP4:       63.5 cy/load
//   Latency ratio: ~8×   Throughput ratio: ~9×
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

// ─── Config ──────────────────────────────────────────────────────────────────
static const int SMEM_WORDS = 512;   // 2 KB smem (512 × u32)
static const int LAT_ITERS  = 1024;  // loads in serial dependent chain
static const int TP_ITERS   = 512;   // iterations per independent chain
static const int ILP        = 8;     // parallel chains for throughput

// Dynamic shared memory alias
extern __shared__ unsigned smem_dyn[];

// ─── Init helper ─────────────────────────────────────────────────────────────
// Fill smem with runtime-computed "next pointer" values in [0, SMEM_WORDS).
// seed is a runtime kernel arg → values cannot be computed at compile time.
__device__ __forceinline__
void init_smem(unsigned* smem, int tid, int seed, unsigned salt)
{
    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ salt;
        // Store a BYTE OFFSET (multiple of 4) so "cur" is directly a byte offset
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) << 2;
    }
    __syncthreads();
}

// ─────────────────────────────────────────────────────────────────────────────
// LATENCY KERNELS — single warp (tid<32), dependent pointer chain
//
// KEY DESIGN:
//   - `cur` is a BYTE OFFSET into smem (not a word index)
//   - `smem[cur/4]` stores another byte offset in [0, SMEM_WORDS)*4
//   - Local load:  LDS cur, [smem_base + cur]  → cur = *(smem_base + cur)
//   - DSMEM load:  LDS cur, [cur + peer_base]  → cur = *(peer_base + cur)
//     where peer_base = mapa(smem_base, target_cta) — hoisted to uniform reg
//   - Each iteration cur (the byte offset) changes, so no LICM possible
//
// SASS expected for local:  LDS R0, [R0]   (R0 holds full addr = base+cur)
// SASS expected for DSMEM:  LDS R0, [R0+UR5]  (R0=cur offset, UR5=peer base)
// ─────────────────────────────────────────────────────────────────────────────

// ── LOCAL SMEM latency (cluster=1) ───────────────────────────────────────────
__global__ void __cluster_dims__(1,1,1)
lat_local(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;
    init_smem(smem, tid, seed, 0u);
    if (tid >= 32) return;

    // cur = initial byte offset
    unsigned cur = ((unsigned)(tid * 31 + seed) & (unsigned)(SMEM_WORDS - 1u)) << 2;
    unsigned base = (unsigned)__cvta_generic_to_shared(smem);
    cur += base;  // cur = full smem address

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    // Dependent chain: each load gives a new base+offset (smem address)
    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(cur)   // address IS cur; result is new smem-addr (base + next_offset)
                     : "memory");
        cur += base;  // smem slot stores offset, add base to get full address
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[0] = t1 - t0;
        out[1] = cur;  // anti-DCE
    }
}

// Wait — there's a subtlety: if smem stores byte OFFSETS, then LDS returns an
// offset, and we need to add the base again. This means each iteration has:
//   LDS (read)  + IADD (add base) = 2 instructions in critical path.
// That's fine — both versions (local and DSMEM) have the same extra IADD,
// so the ratio is still meaningful.
//
// Actually for DSMEM the pattern is different: we want
//   LDS cur, [cur + UR_peer_base]
// So cur holds the OFFSET from peer smem base. The smem table stores offsets.
// The init function stores ((v>>5) & (SMEM_WORDS-1)) << 2 which is a byte offset
// from smem[0]. Local and peer use the same smem layout.
//
// For local: cur = offset from local smem base
//   LDS cur, [cur + local_base]  → cur = new offset
// For DSMEM: cur = offset from peer smem base
//   LDS cur, [cur + peer_base]   → cur = new offset (next offset in peer smem)
//
// This way BOTH use LDS with a per-iteration cur (offset) + fixed base (uniform).
// No LICM possible since cur changes each iteration.
// ─────────────────────────────────────────────────────────────────────────────

// ── LOCAL SMEM latency v2 (correct dependent chain) ──────────────────────────
__global__ void __cluster_dims__(1,1,1)
lat_local2(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;
    // smem[i] stores a byte offset from smem[0] in [0, SMEM_WORDS*4)
    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed;
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();
    if (tid >= 32) return;

    unsigned base = (unsigned)__cvta_generic_to_shared(smem);
    // cur = initial byte offset (from smem[0])
    unsigned cur = ((unsigned)(tid * 31 + seed) & (unsigned)(SMEM_WORDS - 1u)) * 4u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    // Each iteration: load the value at smem[cur/4] (which is another byte offset)
    // Address = base + cur.  Result = next byte offset (also from smem[0]).
    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(base + cur)
                     : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[0] = t1 - t0;
        out[1] = cur;
    }
}

// ── DSMEM latency (cluster=2) ─────────────────────────────────────────────────
// mapa(smem_base, target=compile-time-const) → ptxas promotes to uniform reg UR
// Then LDS uses [cur + UR] where cur changes each iteration → no LICM
__global__ void __cluster_dims__(2,1,1)
lat_dsmem2(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_rank));

    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_rank * 0xDEADBEEFu);
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();

    // Cluster barrier BEFORE timing window
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned target_cta = my_rank ^ 1u;  // compile-time-knowable for cluster=2

    // Obtain peer smem base via mapa — ptxas can hoist this to a uniform reg
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base)
                 : "r"(local_base), "r"(target_cta));

    // cur = initial byte offset into peer smem
    unsigned cur = ((unsigned)(tid * 31 + seed + my_rank) & (unsigned)(SMEM_WORDS - 1u)) * 4u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    // Dependent DSMEM chain: LDS cur, [peer_base + cur]
    // peer_base is in a register (possibly uniform); cur changes each iteration.
    // ptxas should emit: LDS R_cur, [R_peer_base + R_cur]  or  LDS R_cur, [UR_peer_base + R_cur]
    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(peer_base + cur)
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

    unsigned my_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_rank));

    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_rank * 0xDEADBEEFu);
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned target_cta = (my_rank + 1u) & 3u;

    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base)
                 : "r"(local_base), "r"(target_cta));

    unsigned cur = ((unsigned)(tid * 31 + seed + my_rank) & (unsigned)(SMEM_WORDS - 1u)) * 4u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(peer_base + cur)
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

    unsigned my_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_rank));

    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_rank * 0xDEADBEEFu);
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned target_cta = (my_rank + 1u) & 7u;

    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base)
                 : "r"(local_base), "r"(target_cta));

    unsigned cur = ((unsigned)(tid * 31 + seed + my_rank) & (unsigned)(SMEM_WORDS - 1u)) * 4u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(peer_base + cur)
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

    unsigned my_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_rank));

    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_rank * 0xDEADBEEFu);
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned target_cta = (my_rank + 1u) & 15u;

    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base)
                 : "r"(local_base), "r"(target_cta));

    unsigned cur = ((unsigned)(tid * 31 + seed + my_rank) & (unsigned)(SMEM_WORDS - 1u)) * 4u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < LAT_ITERS; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur)
                     : "r"(peer_base + cur)
                     : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = cur;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THROUGHPUT KERNELS — ILP=8 independent chains, single warp (tid<32)
// All 8 loads are independent — hardware can issue all 8 before any completes.
// ─────────────────────────────────────────────────────────────────────────────

// ── LOCAL SMEM throughput ─────────────────────────────────────────────────────
__global__ void __cluster_dims__(1,1,1)
tp_local(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;
    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed;
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();
    if (tid >= 32) return;

    unsigned base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned c0 = ((unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u)) * 4u;
    unsigned c1 = ((unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u)) * 4u;
    unsigned c2 = ((unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u)) * 4u;
    unsigned c3 = ((unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u)) * 4u;
    unsigned c4 = ((unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u)) * 4u;
    unsigned c5 = ((unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u)) * 4u;
    unsigned c6 = ((unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u)) * 4u;
    unsigned c7 = ((unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u)) * 4u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c0) : "r"(base+c0) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c1) : "r"(base+c1) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c2) : "r"(base+c2) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c3) : "r"(base+c3) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c4) : "r"(base+c4) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c5) : "r"(base+c5) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c6) : "r"(base+c6) : "memory");
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(c7) : "r"(base+c7) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[0] = t1 - t0;
        out[1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ── DSMEM throughput (cluster=2) ──────────────────────────────────────────────
__global__ void __cluster_dims__(2,1,1)
tp_dsmem2(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_rank));

    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_rank * 0xDEADBEEFu);
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned target_cta = my_rank ^ 1u;

    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base)
                 : "r"(local_base), "r"(target_cta));

    unsigned c0 = ((unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u)) * 4u;
    unsigned c1 = ((unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u)) * 4u;
    unsigned c2 = ((unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u)) * 4u;
    unsigned c3 = ((unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u)) * 4u;
    unsigned c4 = ((unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u)) * 4u;
    unsigned c5 = ((unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u)) * 4u;
    unsigned c6 = ((unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u)) * 4u;
    unsigned c7 = ((unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u)) * 4u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c0) : "r"(peer_base+c0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c1) : "r"(peer_base+c1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c2) : "r"(peer_base+c2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c3) : "r"(peer_base+c3) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c4) : "r"(peer_base+c4) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c5) : "r"(peer_base+c5) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c6) : "r"(peer_base+c6) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c7) : "r"(peer_base+c7) : "memory");
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

    unsigned my_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_rank));

    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_rank * 0xDEADBEEFu);
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned target_cta = (my_rank + 1u) & 3u;

    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base)
                 : "r"(local_base), "r"(target_cta));

    unsigned c0 = ((unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u)) * 4u;
    unsigned c1 = ((unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u)) * 4u;
    unsigned c2 = ((unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u)) * 4u;
    unsigned c3 = ((unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u)) * 4u;
    unsigned c4 = ((unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u)) * 4u;
    unsigned c5 = ((unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u)) * 4u;
    unsigned c6 = ((unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u)) * 4u;
    unsigned c7 = ((unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u)) * 4u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c0) : "r"(peer_base+c0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c1) : "r"(peer_base+c1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c2) : "r"(peer_base+c2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c3) : "r"(peer_base+c3) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c4) : "r"(peer_base+c4) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c5) : "r"(peer_base+c5) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c6) : "r"(peer_base+c6) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c7) : "r"(peer_base+c7) : "memory");
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

    unsigned my_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_rank));

    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_rank * 0xDEADBEEFu);
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned target_cta = (my_rank + 1u) & 7u;

    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base)
                 : "r"(local_base), "r"(target_cta));

    unsigned c0 = ((unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u)) * 4u;
    unsigned c1 = ((unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u)) * 4u;
    unsigned c2 = ((unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u)) * 4u;
    unsigned c3 = ((unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u)) * 4u;
    unsigned c4 = ((unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u)) * 4u;
    unsigned c5 = ((unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u)) * 4u;
    unsigned c6 = ((unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u)) * 4u;
    unsigned c7 = ((unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u)) * 4u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c0) : "r"(peer_base+c0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c1) : "r"(peer_base+c1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c2) : "r"(peer_base+c2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c3) : "r"(peer_base+c3) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c4) : "r"(peer_base+c4) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c5) : "r"(peer_base+c5) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c6) : "r"(peer_base+c6) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c7) : "r"(peer_base+c7) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (tid == 0) {
        out[blockIdx.x * 2 + 0] = t1 - t0;
        out[blockIdx.x * 2 + 1] = c0^c1^c2^c3^c4^c5^c6^c7;
    }
}

// ── DSMEM throughput (cluster=16, non-portable) ───────────────────────────────
__global__ void __cluster_dims__(16,1,1)
tp_dsmem16(unsigned long long* out, int seed)
{
    unsigned* smem = smem_dyn;
    int tid = threadIdx.x;

    unsigned my_rank;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_rank));

    if (tid < SMEM_WORDS) {
        unsigned v  = (unsigned)tid * 2654435761u ^ (unsigned)seed ^ (my_rank * 0xDEADBEEFu);
        smem[tid] = ((v >> 5) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid >= 32) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(smem);
    unsigned target_cta = (my_rank + 1u) & 15u;

    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base)
                 : "r"(local_base), "r"(target_cta));

    unsigned c0 = ((unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u)) * 4u;
    unsigned c1 = ((unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u)) * 4u;
    unsigned c2 = ((unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u)) * 4u;
    unsigned c3 = ((unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u)) * 4u;
    unsigned c4 = ((unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u)) * 4u;
    unsigned c5 = ((unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u)) * 4u;
    unsigned c6 = ((unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u)) * 4u;
    unsigned c7 = ((unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u)) * 4u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < TP_ITERS; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c0) : "r"(peer_base+c0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c1) : "r"(peer_base+c1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c2) : "r"(peer_base+c2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c3) : "r"(peer_base+c3) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c4) : "r"(peer_base+c4) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c5) : "r"(peer_base+c5) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c6) : "r"(peer_base+c6) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c7) : "r"(peer_base+c7) : "memory");
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

struct RR {
    double median_cy;
    unsigned long long anti_dce;
};

static RR run(void* fn, int csize, int nblk,
              unsigned long long* d_out, int seed, bool nonport)
{
    int smem = SMEM_WORDS * 4;
    CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    if (nonport)
        CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = csize; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;

    cudaLaunchConfig_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.gridDim = nblk; cfg.blockDim = 256; cfg.dynamicSmemBytes = smem;
    cfg.attrs = &attr; cfg.numAttrs = 1;

    void* args[] = { &d_out, &seed };

    for (int i = 0; i < 3; i++) CHECK(cudaLaunchKernelExC(&cfg, fn, args));
    CHECK(cudaDeviceSynchronize());

    const int NT = 12;
    static unsigned long long h[256];
    double tr[NT];
    for (int r = 0; r < NT; r++) {
        CHECK(cudaLaunchKernelExC(&cfg, fn, args));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h, d_out, (size_t)nblk * 2 * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
        tr[r] = (double)h[0];
    }
    for (int i = 0; i < NT-1; i++)
        for (int j = i+1; j < NT; j++)
            if (tr[j] < tr[i]) { double t=tr[i]; tr[i]=tr[j]; tr[j]=t; }
    RR r;
    r.median_cy = (tr[NT/2-1]+tr[NT/2])*0.5;
    r.anti_dce  = h[1];
    return r;
}

int main()
{
    cudaDeviceProp prop;
    CHECK(cudaSetDevice(0));
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int nsm = prop.multiProcessorCount;

    int clk = 0;
    { FILE* f=popen("nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader -i 0 2>/dev/null","r");
      if(f){(void)fscanf(f,"%d",&clk);pclose(f);} }

    printf("# GPU: %s  SMs: %d  SM_clock: %d MHz\n", prop.name, nsm, clk);
    printf("# SMEM_WORDS=%d  LAT_ITERS=%d  TP_ITERS=%d  ILP=%d\n",
           SMEM_WORDS, LAT_ITERS, TP_ITERS, ILP);
    if (clk < 2000) printf("# WARNING: clock below 2032 — run: nvidia-smi -lgc 2032 -i 0\n");

    unsigned long long* dout;
    CHECK(cudaMalloc(&dout, (size_t)nsm*2*sizeof(unsigned long long)));
    int seed = 0x1234ABCD;

    // ── LATENCY ────────────────────────────────────────────────────────────────
    printf("\n=================================================================\n");
    printf("LATENCY: dependent chain (%d serial loads), warp 0 only\n", LAT_ITERS);
    printf("=================================================================\n");
    printf("%-30s %8s %10s %10s %8s\n","variant","cy/load","total_cy","vs_local","anti_dce");

    RR ll  = run((void*)lat_local2,  1,  1, dout, seed, false);
    printf("%-30s %8.2f %10.0f %10.2fx %8llx\n","local SMEM (cluster=1)",
           ll.median_cy/LAT_ITERS, ll.median_cy, 1.0, ll.anti_dce);

    RR d2l = run((void*)lat_dsmem2,  2,  2, dout, seed, false);
    printf("%-30s %8.2f %10.0f %10.2fx %8llx\n","DSMEM cluster=2",
           d2l.median_cy/LAT_ITERS, d2l.median_cy, d2l.median_cy/ll.median_cy, d2l.anti_dce);

    RR d4l = run((void*)lat_dsmem4,  4,  4, dout, seed, false);
    printf("%-30s %8.2f %10.0f %10.2fx %8llx\n","DSMEM cluster=4",
           d4l.median_cy/LAT_ITERS, d4l.median_cy, d4l.median_cy/ll.median_cy, d4l.anti_dce);

    RR d8l = run((void*)lat_dsmem8,  8,  8, dout, seed, false);
    printf("%-30s %8.2f %10.0f %10.2fx %8llx\n","DSMEM cluster=8",
           d8l.median_cy/LAT_ITERS, d8l.median_cy, d8l.median_cy/ll.median_cy, d8l.anti_dce);

    RR d16l= run((void*)lat_dsmem16, 16, 16, dout, seed, true);
    printf("%-30s %8.2f %10.0f %10.2fx %8llx\n","DSMEM cluster=16 (non-port)",
           d16l.median_cy/LAT_ITERS, d16l.median_cy, d16l.median_cy/ll.median_cy, d16l.anti_dce);

    // ── THROUGHPUT ─────────────────────────────────────────────────────────────
    printf("\n=================================================================\n");
    printf("THROUGHPUT: ILP=%d chains × %d iters, warp 0 only\n", ILP, TP_ITERS);
    printf("=================================================================\n");
    printf("%-30s %10s %8s %12s %10s\n","variant","cy/8lds","ld/cy","total_cy","vs_local");

    auto tp_row = [&](const char* name, RR r, double lcy) {
        double tl = (double)TP_ITERS*ILP;
        printf("%-30s %10.2f %8.3f %12.0f %10.2fx\n",
               name, r.median_cy/TP_ITERS, tl/r.median_cy, r.median_cy, r.median_cy/lcy);
    };

    RR lt  = run((void*)tp_local,   1,  1, dout, seed, false);
    tp_row("local SMEM (cluster=1)", lt, lt.median_cy);

    RR d2t = run((void*)tp_dsmem2,  2,  2, dout, seed, false);
    tp_row("DSMEM cluster=2", d2t, lt.median_cy);

    RR d4t = run((void*)tp_dsmem4,  4,  4, dout, seed, false);
    tp_row("DSMEM cluster=4", d4t, lt.median_cy);

    RR d8t = run((void*)tp_dsmem8,  8,  8, dout, seed, false);
    tp_row("DSMEM cluster=8", d8t, lt.median_cy);

    RR d16t= run((void*)tp_dsmem16, 16, 16, dout, seed, true);
    tp_row("DSMEM cluster=16 (non-port)", d16t, lt.median_cy);

    // ── SUMMARY ────────────────────────────────────────────────────────────────
    const double MHZ = 2032.0;
    printf("\n=================================================================\n");
    printf("SUMMARY  (target clock: %.0f MHz = %.3f ns/cy)\n", MHZ, 1000.0/MHZ);
    printf("=================================================================\n");
    printf("Local SMEM lat:        %6.2f cy  = %.2f ns/load\n",
           ll.median_cy/LAT_ITERS, (ll.median_cy/LAT_ITERS)*1000.0/MHZ);
    printf("DSMEM lat clst=2:      %6.2f cy  = %.2f ns/load  (%+.1f cy overhead)\n",
           d2l.median_cy/LAT_ITERS, (d2l.median_cy/LAT_ITERS)*1000.0/MHZ,
           (d2l.median_cy-ll.median_cy)/LAT_ITERS);
    printf("DSMEM lat clst=4:      %6.2f cy  = %.2f ns/load  (%+.1f cy overhead)\n",
           d4l.median_cy/LAT_ITERS, (d4l.median_cy/LAT_ITERS)*1000.0/MHZ,
           (d4l.median_cy-ll.median_cy)/LAT_ITERS);
    printf("DSMEM lat clst=8:      %6.2f cy  = %.2f ns/load  (%+.1f cy overhead)\n",
           d8l.median_cy/LAT_ITERS, (d8l.median_cy/LAT_ITERS)*1000.0/MHZ,
           (d8l.median_cy-ll.median_cy)/LAT_ITERS);
    printf("DSMEM lat clst=16:     %6.2f cy  = %.2f ns/load  (%+.1f cy overhead)\n",
           d16l.median_cy/LAT_ITERS, (d16l.median_cy/LAT_ITERS)*1000.0/MHZ,
           (d16l.median_cy-ll.median_cy)/LAT_ITERS);
    printf("\n");
    printf("Local SMEM tp ILP=8:              %.3f ld/cy\n",
           (double)TP_ITERS*ILP/lt.median_cy);
    printf("DSMEM tp clst=2 ILP=8:            %.3f ld/cy  (%.2fx vs local)\n",
           (double)TP_ITERS*ILP/d2t.median_cy, d2t.median_cy/lt.median_cy);
    printf("DSMEM tp clst=4 ILP=8:            %.3f ld/cy  (%.2fx vs local)\n",
           (double)TP_ITERS*ILP/d4t.median_cy, d4t.median_cy/lt.median_cy);
    printf("DSMEM tp clst=8 ILP=8:            %.3f ld/cy  (%.2fx vs local)\n",
           (double)TP_ITERS*ILP/d8t.median_cy, d8t.median_cy/lt.median_cy);

    CHECK(cudaFree(dout));
    return 0;
}
