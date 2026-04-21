// V19: Hot-spot DSMEM — all N senders reading SAME target CTA
// Plus writes and atomics instruction comparison
//
// Contrast with V18 ring: each CTA gets its own target.
// Here: test if peer SM has hot-spot backpressure.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 8
#define CL 5

// Hot-spot read: all N CTAs read CTA HOT_DST
template<int N_ACTIVE, int HOT_DST, int ILP>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void hot_read(unsigned long long* out, unsigned seed) {
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
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"((unsigned)HOT_DST));

    unsigned c[8];
    #pragma unroll
    for (int k = 0; k < ILP; k++) c[k] = (tid * 4u + k * 32u) & (SMEM_W*4 - 1);

    // N_ACTIVE CTAs active, but skip the hot target itself
    bool active = (my_cta < N_ACTIVE && my_cta != HOT_DST);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    if (active) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                asm volatile("ld.shared::cluster.u32 %0, [%1];"
                             : "=r"(c[k]) : "r"(peer_base + c[k]) : "memory");
            }
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    // Any non-HOT CTA can write timing; use CTA 0 if possible, else CTA 1
    unsigned reporter = (HOT_DST == 0) ? 1 : 0;
    if (my_cta == reporter && tid == 0) {
        out[0] = t1 - t0;
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= c[k];
        ((unsigned*)out)[2] = acc;
    }
}

// Ring write: each CTA writes to (my+1)%CX peer
template<int N_ACTIVE, int ILP>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void ring_write(unsigned long long* out, unsigned seed) {
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

    unsigned a[8];
    #pragma unroll
    for (int k = 0; k < ILP; k++) a[k] = peer_base + (tid * 4u + k * 32u) % (SMEM_W*4);

    unsigned v = seed + tid;
    bool active = (my_cta < N_ACTIVE);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    if (active) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                asm volatile("st.shared::cluster.u32 [%0], %1;"
                             :: "r"(a[k]), "r"(v + i + k) : "memory");
            }
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

// Ring atomic: each CTA does atomic add on (my+1)%CX peer
template<int N_ACTIVE, int ILP>
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

    unsigned a[8];
    #pragma unroll
    for (int k = 0; k < ILP; k++) a[k] = peer_base + (tid * 4u + k * 32u) % (SMEM_W*4);

    unsigned v = seed + tid;
    unsigned acc[8] = {0};
    bool active = (my_cta < N_ACTIVE);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    if (active) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                asm volatile("atom.add.shared::cluster.u32 %0, [%1], %2;"
                             : "=r"(acc[k]) : "r"(a[k]), "r"(v + i + k) : "memory");
            }
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        unsigned s = 0;
        for (int k = 0; k < ILP; k++) s ^= acc[k];
        ((unsigned*)out)[2] = s;
    }
}

template<typename K>
static int avg_cy(K kernel, int N, double* cy_out) {
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

    unsigned long long* d_out;
    cudaMalloc(&d_out, 16);
    double sum = 0; int got = 0;
    for (int r = 0; r < N * 2 && got < N; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, d_out, 42u + r);
        if (e) { cudaGetLastError(); continue; }
        e = cudaDeviceSynchronize();
        if (e) { cudaGetLastError(); continue; }
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        sum += (double)cy;
        got++;
    }
    cudaFree(d_out);
    if (got > 0) { *cy_out = sum / got; return got; }
    return 0;
}

int main() {
    CK(cudaSetDevice(0));
    const double CLOCK_GHZ = 1.920;
    double cy;

    printf("=== V19 DSMEM hot-spot + R/W/atom (CL=5, CX=8, 1920 MHz) ===\n\n");

    // A) Hot-spot reads — all N-1 senders read CTA 0 (HOT)
    printf("A) HOT-SPOT reads: all N senders read same CTA 0 (ILP=4, 32t):\n");
    printf("  N_active  cy_per_CTA  cy/load  BW_agg(GB/s)   Slowdown\n");
    double cy_ref = 0;
    #define HOT_READ(N) { \
        int got = avg_cy(hot_read<N, 0, 4>, 20, &cy); \
        if (got > 0) { \
            double loads_per = 32.0 * 4.0 * CL; \
            double bytes = loads_per * (N-1) * 4.0; \
            double t = cy / CLOCK_GHZ / 1e9; \
            if (N == 2) cy_ref = cy; \
            printf("  %d         %8.0f   %5.2f    %8.2f       %.2fx\n", \
                   N, cy, cy/loads_per, bytes/t/1e9, cy/cy_ref); \
        } }
    HOT_READ(2); HOT_READ(3); HOT_READ(4); HOT_READ(5); HOT_READ(6); HOT_READ(7); HOT_READ(8);
    #undef HOT_READ

    // B) Writes (ring, scaling N)
    printf("\nB) Ring WRITES (ILP=4, 32t × 4 × CL stores per active CTA):\n");
    printf("  N_active  cy_per_CTA  cy/store  BW_agg(GB/s)   Slowdown\n");
    double cy_wr1 = 0;
    #define WR(N) { \
        int got = avg_cy(ring_write<N, 4>, 20, &cy); \
        if (got > 0) { \
            double ops = 32.0 * 4.0 * CL; \
            double bytes = ops * N * 4.0; \
            double t = cy / CLOCK_GHZ / 1e9; \
            if (N == 1) cy_wr1 = cy; \
            printf("  %d         %8.0f   %5.2f     %8.2f       %.2fx\n", \
                   N, cy, cy/ops, bytes/t/1e9, cy/cy_wr1); \
        } }
    WR(1); WR(2); WR(3); WR(4); WR(5); WR(6); WR(7); WR(8);
    #undef WR

    // C) Atomics (ring, scaling N)
    printf("\nC) Ring ATOMICS (atom.add.shared::cluster, ILP=4, 32t):\n");
    printf("  N_active  cy_per_CTA  cy/atom  BW_agg(Gops)   Slowdown\n");
    double cy_at1 = 0;
    #define AT(N) { \
        int got = avg_cy(ring_atom<N, 4>, 20, &cy); \
        if (got > 0) { \
            double ops = 32.0 * 4.0 * CL; \
            double t = cy / CLOCK_GHZ / 1e9; \
            if (N == 1) cy_at1 = cy; \
            printf("  %d         %8.0f   %5.2f     %8.2f       %.2fx\n", \
                   N, cy, cy/ops, ops * N / t / 1e9, cy/cy_at1); \
        } }
    AT(1); AT(2); AT(3); AT(4); AT(5); AT(6); AT(7); AT(8);
    #undef AT

    // D) Atomic ILP sweep at N=8
    printf("\nD) Atomic ILP sweep, N=8:\n");
    int got = avg_cy(ring_atom<8, 1>, 20, &cy);
    if (got > 0) printf("  ILP=1: %.0f cy (%.2f cy/atom)\n", cy, cy / (32.0 * CL));
    got = avg_cy(ring_atom<8, 2>, 20, &cy);
    if (got > 0) printf("  ILP=2: %.0f cy (%.2f cy/atom)\n", cy, cy / (32.0 * 2 * CL));
    got = avg_cy(ring_atom<8, 4>, 20, &cy);
    if (got > 0) printf("  ILP=4: %.0f cy (%.2f cy/atom)\n", cy, cy / (32.0 * 4 * CL));
    got = avg_cy(ring_atom<8, 8>, 20, &cy);
    if (got > 0) printf("  ILP=8: %.0f cy (%.2f cy/atom)\n", cy, cy / (32.0 * 8 * CL));

    return 0;
}
