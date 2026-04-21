// V28: TMA multicast (cp.async.bulk.tensor.shared::cluster.global)
// This is the "real" DSMEM data-movement primitive on Hopper/Blackwell.
// Test: TMA from global to multicast-filled cluster SMEM.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define CX 8
#define TILE_BYTES 16384  // 16 KB tile per CTA
#define SMEM_WORDS (TILE_BYTES / 4)

// Simple 1D TMA multicast: CTA 0 launches TMA, data lands in ALL cluster CTAs' SMEM
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(128, 1)
void tma_multicast(const float* __restrict__ src, unsigned long long* out, unsigned size_elements) {
    __shared__ __align__(16) float buf[SMEM_WORDS];
    __shared__ __align__(8) unsigned long long mbar;

    int tid = threadIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // CTA 0 issues the multicast TMA (ctaMask=0xff for cx=8)
    if (my_cta == 0 && tid == 0) {
        const uint16_t mask = (uint16_t)((1u << CX) - 1);
        unsigned buf_addr = (unsigned)__cvta_generic_to_shared(&buf[0]);
        unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
        // expect arrival of TILE_BYTES
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                     :: "r"(mbar_addr), "r"(TILE_BYTES) : "memory");
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1], %2, [%3], %4;\n"
            :: "r"(buf_addr), "l"(src), "r"(TILE_BYTES), "r"(mbar_addr), "h"(mask)
            : "memory");
    }

    // All CTAs wait on their local mbarrier
    if (tid == 0) {
        int done = 0; int spin = 0;
        unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
        while (!done && spin < 1000000) {
            asm volatile("{.reg .pred p;\n"
                         "mbarrier.try_wait.shared.b64 p, [%1], 0;\n"
                         "selp.u32 %0, 1, 0, p;}\n"
                         : "=r"(done) : "r"(mbar_addr) : "memory");
            spin++;
        }
    }
    __syncthreads();

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Anti-DCE: write first word of each CTA's buf
    if (tid == 0) ((float*)&out[2 + my_cta])[0] = buf[0];

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

// Alternative: per-CTA non-multicast TMA (each CTA loads its own tile)
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(128, 1)
void tma_per_cta(const float* __restrict__ src, unsigned long long* out, unsigned size_elements) {
    __shared__ __align__(16) float buf[SMEM_WORDS];
    __shared__ __align__(8) unsigned long long mbar;

    int tid = threadIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (tid == 0) {
        unsigned buf_addr = (unsigned)__cvta_generic_to_shared(&buf[0]);
        unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                     :: "r"(mbar_addr), "r"(TILE_BYTES) : "memory");
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1], %2, [%3];\n"
            :: "r"(buf_addr), "l"(src + my_cta * SMEM_WORDS), "r"(TILE_BYTES), "r"(mbar_addr)
            : "memory");
    }

    if (tid == 0) {
        int done = 0; int spin = 0;
        unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
        while (!done && spin < 1000000) {
            asm volatile("{.reg .pred p;\n"
                         "mbarrier.try_wait.shared.b64 p, [%1], 0;\n"
                         "selp.u32 %0, 1, 0, p;}\n"
                         : "=r"(done) : "r"(mbar_addr) : "memory");
            spin++;
        }
    }
    __syncthreads();

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) ((float*)&out[2 + my_cta])[0] = buf[0];

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

int main() {
    CK(cudaSetDevice(0));
    const double CLOCK_GHZ = 1.920;

    // Allocate source
    unsigned N = SMEM_WORDS * CX;
    float* d_src;
    CK(cudaMalloc(&d_src, N * 4));
    float* h_src = (float*)malloc(N * 4);
    for (unsigned i = 0; i < N; i++) h_src[i] = i * 0.1f;
    cudaMemcpy(d_src, h_src, N * 4, cudaMemcpyHostToDevice);

    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CX, 1, 1);
    cfg.blockDim = dim3(128, 1, 1);
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CX;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    printf("=== V28 TMA multicast (cp.async.bulk) — 8 CTAs × 16 KB tile ===\n\n");

    double sum = 0; int got = 0;
    for (int r = 0; r < 15; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, tma_multicast, (const float*)d_src, d_out, N);
        if (e) { cudaGetLastError(); continue; }
        e = cudaDeviceSynchronize();
        if (e) { cudaGetLastError(); continue; }
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        sum += (double)cy;
        got++;
    }
    if (got > 0) {
        double avg = sum / got;
        double t_us = avg / CLOCK_GHZ / 1e3;
        double bytes_cluster = TILE_BYTES * CX;  // multicast delivers to all CX CTAs
        double bw_gbps = bytes_cluster / (avg / CLOCK_GHZ / 1e9) / 1e9;
        printf("A) MULTICAST TMA (8-way): %.0f cy (%.2f us)\n", avg, t_us);
        printf("   Effective BW: %.2f GB/s (%.2f KB × 8 CTAs / %.2f us)\n",
               bw_gbps, TILE_BYTES/1024.0, t_us);
        printf("   Per-CTA amortized: %.2f GB/s\n", bw_gbps / CX);
    } else printf("A) MULTICAST: failed\n");

    sum = 0; got = 0;
    for (int r = 0; r < 15; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, tma_per_cta, (const float*)d_src, d_out, N);
        if (e) { cudaGetLastError(); continue; }
        e = cudaDeviceSynchronize();
        if (e) { cudaGetLastError(); continue; }
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        sum += (double)cy;
        got++;
    }
    if (got > 0) {
        double avg = sum / got;
        double t_us = avg / CLOCK_GHZ / 1e3;
        double bytes_cluster = TILE_BYTES * CX;
        double bw_gbps = bytes_cluster / (avg / CLOCK_GHZ / 1e9) / 1e9;
        printf("\nB) PER-CTA TMA (8 independent): %.0f cy (%.2f us)\n", avg, t_us);
        printf("   Aggregate BW: %.2f GB/s\n", bw_gbps);
    } else printf("\nB) PER-CTA: failed\n");

    free(h_src);
    cudaFree(d_src);
    cudaFree(d_out);
    return 0;
}
