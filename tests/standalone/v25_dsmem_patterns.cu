// V25: Realistic DSMEM patterns
// A) All-to-all read — each CTA reads from ALL N-1 peers simultaneously
// B) Alignment: 4B, 8B, 16B stores
// C) DSMEM aggregated read across many warps for max BW
// D) Async barrier arrival: mbarrier fall-through test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 1024
#define CX 8

// A) All-to-all: each CTA reads from every peer (7 targets)
template<int N_WARP, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(N_WARP*32, 1)
void all2all_full(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += N_WARP*32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    // Compute peer bases array
    unsigned peer_bases[CX];
    #pragma unroll
    for (int p = 0; p < CX; p++) {
        unsigned pb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(pb) : "r"(local_base), "r"(p));
        peer_bases[p] = pb;
    }

    unsigned cur = (tid * 4u) & (SMEM_W*4 - 1);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        #pragma unroll
        for (int p = 0; p < CX; p++) {
            if (p == my_cta) continue;  // skip self
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(peer_bases[p] + cur) : "memory");
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

// B) Aligned store width test
enum StWidth { W_U8=0, W_U16=1, W_U32=2, W_U64=3, W_V2U64=4, W_V4U32=5 };

template<int WIDTH, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void store_width(unsigned long long* out, unsigned seed) {
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

    unsigned addr = peer_base + ((tid * 16u) & (SMEM_W*4 - 16));
    unsigned v = seed + tid;
    unsigned long long v64 = (unsigned long long)v | ((unsigned long long)(v+1) << 32);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        if (WIDTH == W_U32)
            asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(addr), "r"(v + i) : "memory");
        else if (WIDTH == W_U64)
            asm volatile("st.shared::cluster.u64 [%0], %1;" :: "r"(addr), "l"(v64 + i) : "memory");
        else if (WIDTH == W_V2U64)
            asm volatile("st.shared::cluster.v2.u64 [%0], {%1, %2};"
                         :: "r"(addr), "l"(v64 + i), "l"(v64 + i + 1) : "memory");
        else if (WIDTH == W_V4U32)
            asm volatile("st.shared::cluster.v4.u32 [%0], {%1, %2, %3, %4};"
                         :: "r"(addr), "r"(v + i), "r"(v + i + 1), "r"(v + i + 2), "r"(v + i + 3) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

// C) All-reduce pattern via ring + fence + barrier
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void ring_allreduce(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    __shared__ unsigned int recv_buf;

    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) smem[i] = 0;
    if (tid == 0) recv_buf = 0;
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&recv_buf);
    unsigned next_cta = (my_cta + 1u) % CX;
    unsigned peer_recv;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_recv) : "r"(local_base), "r"(next_cta));

    // Simulate: pass my partial through CX-1 hops accumulating
    unsigned payload = seed + my_cta * 100u;
    unsigned acc = payload;

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (tid == 0) {
        #pragma unroll 1
        for (int step = 0; step < CX - 1; step++) {
            // Write my accumulator to next CTA's recv_buf
            asm volatile("st.shared::cluster.u32 [%0], %1;"
                         :: "r"(peer_recv), "r"(acc) : "memory");
            asm volatile("fence.sc.cluster;" ::: "memory");
            asm volatile("barrier.cluster.arrive;" ::: "memory");
            asm volatile("barrier.cluster.wait;"  ::: "memory");
            // Read my recv_buf (now set by previous CTA)
            acc += recv_buf;
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
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

    printf("=== V25 DSMEM patterns (CX=8, 1920 MHz) ===\n\n");

    // A) All-to-all ring
    printf("A) All-to-all reads (each CTA reads CX-1=7 peers in sequence, CL=3):\n");
    printf("  warps   cy_per_CTA   BW_agg(GB/s)\n");
    #define A2A(W) { \
        int got = avg_cy_t(W*32, all2all_full<W, 3>, 12, &cy); \
        if (got > 0) { \
            double loads_per_cta = W * 32.0 * (CX-1) * 3; \
            double t = cy / CLOCK_GHZ / 1e9; \
            double bw = loads_per_cta * CX * 4.0 / t / 1e9; \
            printf("  %d       %8.0f      %.2f\n", W, cy, bw); \
        } }
    A2A(1); A2A(2); A2A(4);
    #undef A2A

    // B) Store width
    printf("\nB) Store width (CTA 0 → CTA 1, 1 thread, CL=50):\n");
    double cy32, cy64;
    int got = avg_cy_t(32, store_width<W_U32, 50>, 20, &cy);
    cy32 = cy;
    if (got > 0) printf("  st.shared::cluster.u32:    %.0f cy (%.2f cy/st, %.2f bytes/cy)\n", cy, cy/50, 4.0*50.0/cy);
    got = avg_cy_t(32, store_width<W_U64, 50>, 20, &cy);
    cy64 = cy;
    if (got > 0) printf("  st.shared::cluster.u64:    %.0f cy (%.2f cy/st, %.2f bytes/cy)\n", cy, cy/50, 8.0*50.0/cy);
    got = avg_cy_t(32, store_width<W_V2U64, 50>, 20, &cy);
    if (got > 0) printf("  st.shared::cluster.v2.u64: %.0f cy (%.2f cy/st, %.2f bytes/cy)\n", cy, cy/50, 16.0*50.0/cy);
    got = avg_cy_t(32, store_width<W_V4U32, 50>, 20, &cy);
    if (got > 0) printf("  st.shared::cluster.v4.u32: %.0f cy (%.2f cy/st, %.2f bytes/cy)\n", cy, cy/50, 16.0*50.0/cy);

    // C) All-reduce ring
    printf("\nC) Ring all-reduce (CX-1=7 steps, each with fence+barrier):\n");
    got = avg_cy_t(32, ring_allreduce<1>, 15, &cy);
    if (got > 0) {
        double t = cy / CLOCK_GHZ / 1e9;
        printf("  Total: %.0f cy (%.2f us), %.1f cy per step\n", cy, t*1e6, cy/(CX-1));
    }

    return 0;
}
