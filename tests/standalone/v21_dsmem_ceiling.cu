// V21: DSMEM peak BW ceiling — push threads × warps × ILP × CTAs simultaneously
// Plus: fence.sc.cluster cost isolated, multicast comparison, read+write overlap

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 1024
#define CX 8

// A) Max BW push: CX CTAs × N warps × ILP reads to ring peer
template<int N_WARP, int ILP, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(N_WARP * 32, 1)
void push_ring_rd(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += N_WARP * 32) {
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

    unsigned c[8];
    #pragma unroll
    for (int k = 0; k < ILP; k++) c[k] = (tid * 4u + k * 32u) & (SMEM_W*4 - 1);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(c[k]) : "r"(peer_base + c[k]) : "memory");
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= c[k];
        ((unsigned*)out)[2] = acc;
    }
}

// B) Push writes: N warps × ILP stores ring
template<int N_WARP, int ILP, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(N_WARP * 32, 1)
void push_ring_wr(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += N_WARP * 32) smem[i] = 0;
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
    for (int k = 0; k < ILP; k++) a[k] = peer_base + ((tid * 4u + k * 32u) % (SMEM_W*4));

    unsigned v = seed + tid;

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            asm volatile("st.shared::cluster.u32 [%0], %1;"
                         :: "r"(a[k]), "r"(v + i + k) : "memory");
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

// C) Fence cost test: compare write loop WITH and WITHOUT fence.sc.cluster
template<int WITH_FENCE, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void fence_cost(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) smem[i] = 0;
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(1u));

    unsigned addr = peer_base + (seed & (SMEM_W*4-4));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    if (my_cta == 0 && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("st.shared::cluster.u32 [%0], %1;"
                         :: "r"(addr), "r"(seed + i) : "memory");
            if (WITH_FENCE) asm volatile("fence.sc.cluster;" ::: "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

// D) Read+Write concurrent: half warps read, half write
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(128, 1)
void rw_concurrent(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 128) smem[i] = (i * 37) & (SMEM_W*4-1);
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"((my_cta + 1u) % CX));

    int warp_id = tid / 32;
    bool is_reader = (warp_id < 2);

    unsigned c[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) c[k] = ((tid * 4u + k * 32u) & (SMEM_W*4 - 1));

    unsigned addr = peer_base + ((tid * 4u) & (SMEM_W*4-4));
    unsigned v = seed + tid;

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        if (is_reader) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                asm volatile("ld.shared::cluster.u32 %0, [%1];"
                             : "=r"(c[k]) : "r"(peer_base + c[k]) : "memory");
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                asm volatile("st.shared::cluster.u32 [%0], %1;"
                             :: "r"(addr + k*32), "r"(v + i + k) : "memory");
            }
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        unsigned acc = c[0] ^ c[1] ^ c[2] ^ c[3];
        ((unsigned*)out)[2] = acc;
    }
}

template<typename K, typename... Args>
static int avg_cy_t(int threads, K kernel, int N, double* cy_out, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CX, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
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
        cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, d_out, args..., 42u + r);
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

    printf("=== V21 DSMEM ceiling + fence + R+W overlap (CX=8, 1920 MHz) ===\n\n");

    // A) Max read BW — push warps × ILP
    printf("A) Max read BW (all 8 CTAs active):\n");
    printf("  warps×ILP  threads/CTA  cy_per_CTA   cy/load    BW_per_CTA   BW_agg\n");
    #define RD(W, I, CL) { \
        int got = avg_cy_t(W*32, push_ring_rd<W, I, CL>, 15, &cy); \
        if (got > 0) { \
            double loads = W * 32.0 * I * CL; \
            double t = cy / CLOCK_GHZ / 1e9; \
            double bw_cta = loads * 4.0 / t / 1e9; \
            printf("  %dx%-3d      %4d         %7.0f     %5.2f     %7.2f      %7.2f GB/s\n", \
                   W, I, W*32, cy, cy/loads, bw_cta, bw_cta * CX); \
        } else printf("  %dx%-3d CRASH\n", W, I); }
    RD(1, 1, 5); RD(1, 4, 5); RD(1, 8, 5);
    RD(2, 4, 5); RD(4, 4, 5); RD(8, 4, 5);
    RD(2, 8, 5); RD(4, 8, 5); RD(8, 8, 5);
    RD(16, 4, 5);
    #undef RD

    // B) Max write BW
    printf("\nB) Max write BW (all 8 CTAs active):\n");
    printf("  warps×ILP  threads/CTA  cy_per_CTA   cy/store   BW_per_CTA   BW_agg\n");
    #define WR(W, I, CL) { \
        int got = avg_cy_t(W*32, push_ring_wr<W, I, CL>, 15, &cy); \
        if (got > 0) { \
            double ops = W * 32.0 * I * CL; \
            double t = cy / CLOCK_GHZ / 1e9; \
            double bw_cta = ops * 4.0 / t / 1e9; \
            printf("  %dx%-3d      %4d         %7.0f     %5.2f     %7.2f      %7.2f GB/s\n", \
                   W, I, W*32, cy, cy/ops, bw_cta, bw_cta * CX); \
        } else printf("  %dx%-3d CRASH\n", W, I); }
    WR(1, 1, 5); WR(1, 4, 5); WR(1, 8, 5);
    WR(2, 4, 5); WR(4, 4, 5); WR(8, 4, 5);
    WR(8, 8, 5);
    WR(16, 4, 5);
    #undef WR

    // C) Fence cost
    printf("\nC) Fence.sc.cluster cost (1 thread, CL writes):\n");
    double cy_nof, cy_f;
    avg_cy_t(32, fence_cost<0, 20>, 20, &cy_nof);
    avg_cy_t(32, fence_cost<1, 20>, 20, &cy_f);
    printf("  No fence,    CL=20: %.0f cy (%.2f cy/st)\n", cy_nof, cy_nof/20.0);
    printf("  With fence, CL=20: %.0f cy (%.2f cy/st_with_fence)\n", cy_f, cy_f/20.0);
    printf("  → fence overhead: %.2f cy per fence\n", (cy_f - cy_nof) / 20.0);

    // D) R+W concurrent
    printf("\nD) R+W concurrent (2 warps read, 2 warps write, CL=5):\n");
    int got = avg_cy_t(128, rw_concurrent<5>, 15, &cy);
    if (got > 0) {
        printf("  %.0f cy total\n", cy);
        double t = cy / CLOCK_GHZ / 1e9;
        double rd_ops = 2 * 32.0 * 4.0 * 5;  // 2 warps × 32 threads × 4 ILP × CL
        double wr_ops = 2 * 32.0 * 4.0 * 5;
        printf("  Read BW (2 warps): %.2f GB/s, write BW (2 warps): %.2f GB/s\n",
               rd_ops * 4.0 / t / 1e9, wr_ops * 4.0 / t / 1e9);
    }

    return 0;
}
