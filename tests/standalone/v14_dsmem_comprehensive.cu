// V14: Comprehensive DSMEM BW characterization
//
// Experiments:
// A) Peer-distance: 1 CTA reads from peer N (N=1..CX-1), only 1 CTA doing loads
// B) Many-to-many: all CTAs read ring (my+1)%CX simultaneously
// C) All-to-all:   all CTAs read all peers concurrently
// D) Instruction sweep: ld vs st vs atom.add
// E) ILP sweep: 1, 2, 4, 8 within single thread / single warp
// F) Thread count sweep: 1 / 8 / 16 / 32 threads per warp
//
// All tests use dep chains or cumulative results for DCE immunity.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256

// A) Peer-distance test: CTA 0 reads from CTA `PEER`, others idle.
//    32 threads × ILP=4 dep chains
template<int CX, int PEER, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void peer_dist_rd(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta != 0) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = PEER;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    unsigned c0 = (tid * 4u + 0u)   & (SMEM_W*4 - 1);
    unsigned c1 = (tid * 4u + 64u)  & (SMEM_W*4 - 1);
    unsigned c2 = (tid * 4u + 128u) & (SMEM_W*4 - 1);
    unsigned c3 = (tid * 4u + 192u) & (SMEM_W*4 - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c0) : "r"(peer_base + c0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c1) : "r"(peer_base + c1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c2) : "r"(peer_base + c2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c3) : "r"(peer_base + c3) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) out[0] = t1 - t0;
    ((unsigned*)out)[2 + tid] = c0 ^ c1 ^ c2 ^ c3;
}

// B) All-CTA ring: every CTA reads from (my+1)%CX
template<int CX, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void ring_rd(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (my_cta + 1u) % CX;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    unsigned c0 = (tid * 4u + 0u)   & (SMEM_W*4 - 1);
    unsigned c1 = (tid * 4u + 64u)  & (SMEM_W*4 - 1);
    unsigned c2 = (tid * 4u + 128u) & (SMEM_W*4 - 1);
    unsigned c3 = (tid * 4u + 192u) & (SMEM_W*4 - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c0) : "r"(peer_base + c0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c1) : "r"(peer_base + c1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c2) : "r"(peer_base + c2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(c3) : "r"(peer_base + c3) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
    if (my_cta == 0) ((unsigned*)out)[2 + tid] = c0 ^ c1 ^ c2 ^ c3;
}

// C) All-to-all: each CTA reads from every other CTA (one load per peer per chain step)
template<int CX, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void all2all_rd(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    // Precompute peer bases for each peer
    unsigned peer_bases[CX];
    #pragma unroll
    for (int p = 0; p < CX; p++) {
        unsigned pb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(pb) : "r"(local_base), "r"(p));
        peer_bases[p] = pb;
    }

    unsigned cur = (tid * 4u) & (SMEM_W*4 - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        #pragma unroll
        for (int p = 0; p < CX; p++) {
            if (p == my_cta) continue;
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(peer_bases[p] + cur) : "memory");
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
    if (my_cta == 0) ((unsigned*)out)[2 + tid] = cur;
}

// D) Write instruction test: st.shared::cluster — can't chain, but we can count instructions per cy
template<int CX, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void ring_wr(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) smem[i] = 0;
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (my_cta + 1u) % CX;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    // Each thread writes into its own slot at varying offsets
    unsigned addr = peer_base + (tid * 4u);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    // Independent writes — can't chain, but loads can't hoist either
    // Use tid-dependent value (prevents constant-folding)
    unsigned v = seed + tid;
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(addr), "r"(v + i) : "memory");
        asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(addr + 64u), "r"(v + i + 1) : "memory");
        asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(addr + 128u), "r"(v + i + 2) : "memory");
        asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(addr + 192u), "r"(v + i + 3) : "memory");
    }
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

// E) Atomic: atom.add.shared::cluster
template<int CX, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void ring_atom(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) smem[i] = 0;
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (my_cta + 1u) % CX;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    unsigned addr = peer_base + (tid * 4u);
    unsigned v = seed + tid;
    unsigned acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("atom.add.shared::cluster.u32 %0, [%1], %2;"
                     : "=r"(acc) : "r"(addr), "r"(v + i) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
    if (my_cta == 0) ((unsigned*)out)[2 + tid] = acc;
}

static void stats(const double* s, int n, double& mn, double& avg, double& mx) {
    mn = 1e30; mx = 0; avg = 0;
    for (int i = 0; i < n; i++) { if (s[i]<mn) mn=s[i]; if (s[i]>mx) mx=s[i]; avg+=s[i]; }
    avg /= n;
}

template<typename K>
static int run_sample(K kernel, int CX, unsigned long long* d_out, double* cy_out, int N, const char* label) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CX, 1, 1);
    cfg.blockDim = dim3(32, 1, 1);
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CX;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    int got = 0, crashes = 0;
    for (int r = 0; r < N * 2 && got < N; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, d_out, 42u + r);
        if (e) { crashes++; continue; }
        e = cudaDeviceSynchronize();
        if (e) { crashes++; continue; }
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        cy_out[got++] = (double)cy;
    }
    if (crashes > N/4) fprintf(stderr, "  %s: %d crashes\n", label, crashes);
    return (got >= N/2) ? got : 0;
}

int main() {
    CK(cudaSetDevice(0));
    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));
    const double CLOCK_GHZ = 1.920;
    double cy[64], mn, avg, mx;

    printf("=== V14 DSMEM comprehensive (1920 MHz) ===\n\n");

    // A) PEER DISTANCE at CX=8 — one CTA reading from peer 1..7
    printf("A) Peer distance at CX=8 (32t, ILP=4, CL=5, CTA 0 only active):\n");
    {
        int n;
        #define RUN_PEER(P) { \
            n = run_sample(peer_dist_rd<8, P, 5>, 8, d_out, cy, 20, "peer" #P); \
            if (n >= 10) { stats(cy, n, mn, avg, mx); \
                double loads = 32.0 * 4.0 * 5.0; \
                double bytes = loads * 4.0; \
                double time_s = avg / CLOCK_GHZ / 1e9; \
                printf("  peer=%d: avg %.0f cy (%.1f cy/load), time %.3f us, BW %.2f GB/s\n", \
                       P, avg, avg/loads, time_s*1e6, bytes/time_s/1e9); \
            } else { printf("  peer=%d: crashed too much\n", P); } }
        RUN_PEER(1); RUN_PEER(2); RUN_PEER(3); RUN_PEER(4);
        RUN_PEER(5); RUN_PEER(6); RUN_PEER(7);
        #undef RUN_PEER
    }

    // B) Ring — all CTAs read simultaneously at each cluster size
    printf("\nB) Ring all-CTAs-active (32t, ILP=4, CL=5):\n");
    {
        int n;
        #define RUN_RING(CX) { \
            n = run_sample(ring_rd<CX, 5>, CX, d_out, cy, 20, "ring" #CX); \
            if (n >= 10) { stats(cy, n, mn, avg, mx); \
                double loads_per_cta = 32.0 * 4.0 * 5.0; \
                double total_loads = loads_per_cta * CX; \
                double bytes = total_loads * 4.0; \
                double time_s = avg / CLOCK_GHZ / 1e9; \
                printf("  cx=%d: avg %.0f cy (%.1f cy/load per CTA), aggregate BW %.2f GB/s (%.1f GB/s per CTA)\n", \
                       CX, avg, avg/loads_per_cta, bytes/time_s/1e9, bytes/time_s/1e9/CX); \
            } else { printf("  cx=%d: crashed\n", CX); } }
        RUN_RING(2); RUN_RING(3); RUN_RING(4); RUN_RING(5);
        RUN_RING(6); RUN_RING(7); RUN_RING(8);
        #undef RUN_RING
    }

    // C) All-to-all — each CTA reads from all other CTAs
    printf("\nC) All-to-all (each CTA reads all peers, 32t, CL=3 small for crash avoidance):\n");
    {
        int n;
        #define RUN_A2A(CX) { \
            n = run_sample(all2all_rd<CX, 3>, CX, d_out, cy, 15, "a2a" #CX); \
            if (n >= 7) { stats(cy, n, mn, avg, mx); \
                double loads_per_cta = 32.0 * (CX-1) * 3.0; \
                double total_loads = loads_per_cta * CX; \
                double bytes = total_loads * 4.0; \
                double time_s = avg / CLOCK_GHZ / 1e9; \
                printf("  cx=%d: avg %.0f cy (%.1f cy/load per CTA), aggregate BW %.2f GB/s\n", \
                       CX, avg, avg/loads_per_cta, bytes/time_s/1e9); \
            } else { printf("  cx=%d: crashed\n", CX); } }
        RUN_A2A(2); RUN_A2A(4); RUN_A2A(8);
        #undef RUN_A2A
    }

    // D) Writes (ring)
    printf("\nD) Ring writes (32t × 4 stores × CL, bar.cluster.arrive at end):\n");
    {
        int n;
        #define RUN_WR(CX, CL) { \
            n = run_sample(ring_wr<CX, CL>, CX, d_out, cy, 20, "wr" #CX); \
            if (n >= 10) { stats(cy, n, mn, avg, mx); \
                double stores_per_cta = 32.0 * 4.0 * CL; \
                double total = stores_per_cta * CX; \
                double bytes = total * 4.0; \
                double time_s = avg / CLOCK_GHZ / 1e9; \
                printf("  cx=%d CL=%d: avg %.0f cy (%.1f cy/st per CTA), aggregate BW %.2f GB/s\n", \
                       CX, CL, avg, avg/stores_per_cta, bytes/time_s/1e9); \
            } else { printf("  cx=%d: crashed\n", CX); } }
        RUN_WR(2, 20); RUN_WR(4, 5); RUN_WR(8, 5);
        #undef RUN_WR
    }

    // E) Atomics (ring)
    printf("\nE) Ring atomic.add (32t × 1 atomic × CL):\n");
    {
        int n;
        #define RUN_AT(CX, CL) { \
            n = run_sample(ring_atom<CX, CL>, CX, d_out, cy, 20, "at" #CX); \
            if (n >= 10) { stats(cy, n, mn, avg, mx); \
                double ops_per_cta = 32.0 * CL; \
                double total = ops_per_cta * CX; \
                double bytes = total * 8.0; /* RMW */ \
                double time_s = avg / CLOCK_GHZ / 1e9; \
                printf("  cx=%d CL=%d: avg %.0f cy (%.1f cy/atom per CTA), aggregate BW %.2f GB/s\n", \
                       CX, CL, avg, avg/ops_per_cta, bytes/time_s/1e9); \
            } else { printf("  cx=%d: crashed\n", CX); } }
        RUN_AT(2, 20); RUN_AT(4, 5); RUN_AT(8, 5);
        #undef RUN_AT
    }

    cudaFree(d_out);
    return 0;
}
