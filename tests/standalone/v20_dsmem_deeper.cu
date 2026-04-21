// V20: Deeper DSMEM coverage
// A) Per-pair write latency (is write also pair-dependent?)
// B) Per-pair atomic latency (single thread, single op, 10 iters)
// C) Multi-warp reads (2/4/8 warps per CTA)
// D) 128-bit loads via ld.shared::cluster.v4.b32

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 8

// A) Pair write latency — 1 thread, 1 store × CL=50 iters
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void pair_write(unsigned long long* out, unsigned src, unsigned dst, unsigned seed) {
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
                 : "=r"(peer_base) : "r"(local_base), "r"(dst));

    unsigned addr = peer_base + (seed & (SMEM_W*4-4));  // avoid LICM via seed
    unsigned long long t0 = 0, t1 = 0;

    if (my_cta == src && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("st.shared::cluster.u32 [%0], %1;"
                         :: "r"(addr), "r"(seed + i) : "memory");
        }
        // fence to measure completion not just issue
        asm volatile("fence.sc.cluster;" ::: "memory");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == src && tid == 0) out[0] = t1 - t0;
}

// B) Pair atomic latency — single thread, 1 atom × CL=50 iters
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void pair_atom(unsigned long long* out, unsigned src, unsigned dst, unsigned seed) {
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
                 : "=r"(peer_base) : "r"(local_base), "r"(dst));

    unsigned addr = peer_base + (seed & (SMEM_W*4-4));
    unsigned acc = 0;
    unsigned long long t0 = 0, t1 = 0;

    if (my_cta == src && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("atom.add.shared::cluster.u32 %0, [%1], %2;"
                         : "=r"(acc) : "r"(addr), "r"(acc + i) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == src && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = acc;
    }
}

// C) Multi-warp ring read — single CTA with N_WARP warps each with own chain
template<int N_WARP, int ILP, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(N_WARP * 32, 1)
void multiwarp_ring(unsigned long long* out, unsigned seed) {
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

// D) 128-bit load ring
template<int ILP, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void ring_ld128(unsigned long long* out, unsigned seed) {
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

    // For 128-bit: alignment-aligned addresses
    unsigned c[8];
    #pragma unroll
    for (int k = 0; k < ILP; k++) c[k] = (tid * 16u + k * 32u) & (SMEM_W*4 - 16);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned x0[8], x1[8], x2[8], x3[8];
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(x0[k]), "=r"(x1[k]), "=r"(x2[k]), "=r"(x3[k])
                         : "r"(peer_base + c[k]) : "memory");
            c[k] = x0[k] & (SMEM_W*4 - 16);  // re-derive next address to keep chain
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= x0[k] ^ x1[k] ^ x2[k] ^ x3[k];
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

    printf("=== V20 DSMEM deeper (CX=8, 1920 MHz) ===\n\n");

    // A) Per-pair WRITE latency (1 thread, fence.sc.cluster for completion)
    printf("A) Pair write latency (1 thread, CL=50, fence.sc.cluster):\n");
    printf("       ");
    for (int d = 0; d < CX; d++) printf("  D=%d  ", d);
    printf("\n");
    for (int s = 0; s < CX; s++) {
        printf("  S=%d ", s);
        for (int d = 0; d < CX; d++) {
            if (s == d) { printf("  --   "); continue; }
            int got = avg_cy_t(32, pair_write<50>, 15, &cy, (unsigned)s, (unsigned)d);
            if (got > 0) printf(" %5.1f ", cy / 50.0);
            else printf("  ?    ");
        }
        printf("\n");
    }

    // B) Per-pair ATOMIC latency (1 thread, CL=20)
    printf("\nB) Pair atomic latency (1 thread, CL=20):\n");
    printf("       ");
    for (int d = 0; d < CX; d++) printf("  D=%d  ", d);
    printf("\n");
    for (int s = 0; s < CX; s++) {
        printf("  S=%d ", s);
        for (int d = 0; d < CX; d++) {
            if (s == d) { printf("  --   "); continue; }
            int got = avg_cy_t(32, pair_atom<20>, 15, &cy, (unsigned)s, (unsigned)d);
            if (got > 0) printf(" %5.1f ", cy / 20.0);
            else printf("  ?    ");
        }
        printf("\n");
    }

    // C) Multi-warp ring reads at ILP=4, CL=5
    printf("\nC) Multi-warp ring reads (ILP=4, CL=5, ring my+1):\n");
    printf("  1 warp:  ");
    int got = avg_cy_t(32, multiwarp_ring<1, 4, 5>, 15, &cy);
    if (got > 0) {
        double loads = 32.0 * 4.0 * 5.0;
        double t = cy / CLOCK_GHZ / 1e9;
        printf("%.0f cy (%.2f cy/load), per-CTA BW %.2f GB/s\n",
               cy, cy/loads, loads * 4.0 / t / 1e9);
    }
    printf("  2 warps: ");
    got = avg_cy_t(64, multiwarp_ring<2, 4, 5>, 15, &cy);
    if (got > 0) {
        double loads = 64.0 * 4.0 * 5.0;
        double t = cy / CLOCK_GHZ / 1e9;
        printf("%.0f cy (%.2f cy/load), per-CTA BW %.2f GB/s\n",
               cy, cy/loads, loads * 4.0 / t / 1e9);
    }
    printf("  4 warps: ");
    got = avg_cy_t(128, multiwarp_ring<4, 4, 5>, 15, &cy);
    if (got > 0) {
        double loads = 128.0 * 4.0 * 5.0;
        double t = cy / CLOCK_GHZ / 1e9;
        printf("%.0f cy (%.2f cy/load), per-CTA BW %.2f GB/s\n",
               cy, cy/loads, loads * 4.0 / t / 1e9);
    }
    printf("  8 warps: ");
    got = avg_cy_t(256, multiwarp_ring<8, 4, 5>, 15, &cy);
    if (got > 0) {
        double loads = 256.0 * 4.0 * 5.0;
        double t = cy / CLOCK_GHZ / 1e9;
        printf("%.0f cy (%.2f cy/load), per-CTA BW %.2f GB/s\n",
               cy, cy/loads, loads * 4.0 / t / 1e9);
    }

    // D) 128-bit vs 32-bit loads
    printf("\nD) 128-bit loads (ld.shared::cluster.v4.b32):\n");
    got = avg_cy_t(32, ring_ld128<1, 5>, 15, &cy);
    if (got > 0) {
        double loads = 32.0 * 5.0;
        double t = cy / CLOCK_GHZ / 1e9;
        printf("  ILP=1: %.0f cy (%.2f cy/128b-load), per-CTA BW %.2f GB/s\n",
               cy, cy/loads, loads * 16.0 / t / 1e9);
    }
    got = avg_cy_t(32, ring_ld128<4, 5>, 15, &cy);
    if (got > 0) {
        double loads = 32.0 * 4 * 5.0;
        double t = cy / CLOCK_GHZ / 1e9;
        printf("  ILP=4: %.0f cy (%.2f cy/128b-load), per-CTA BW %.2f GB/s\n",
               cy, cy/loads, loads * 16.0 / t / 1e9);
    }

    return 0;
}
