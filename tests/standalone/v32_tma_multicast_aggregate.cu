// V32: Aggregate TMA multicast ceiling across ALL clusters
// 10-rule rigor: theoretical 58.4 TB/s effective (7.3 TB/s HBM × 8 multicast)
//
// Launch 18 clusters × 8 CTAs = 144 CTAs. Each cluster does N TMA multicast
// iterations with 32 KB tile. Measure aggregate effective BW.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define CX 8
#define TILE_BYTES 65536
#define SMEM_BYTES 131072  // double buffer (2× tile)
#define NUM_TMAS_PER_ITER 2

template<int N_ITERS>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(128, 1)
void tma_mcast_persistent(const float* src, unsigned long long* out, unsigned src_words) {
    extern __shared__ __align__(16) char buf[];
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

    unsigned buf_addr = (unsigned)__cvta_generic_to_shared(&buf[0]);
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
    const uint16_t mask = (uint16_t)((1u << CX) - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        if (tid == 0) {
            asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                         :: "r"(mbar_addr) : "memory");
        }
        __syncthreads();
        // Expect N TMAs worth of bytes
        if (tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                         :: "r"(mbar_addr), "r"(TILE_BYTES * NUM_TMAS_PER_ITER) : "memory");
        }
        __syncthreads();
        // CTA 0 issues NUM_TMAS_PER_ITER multicast TMAs to different buffer slots
        if (my_cta == 0 && tid == 0) {
            #pragma unroll
            for (int t = 0; t < NUM_TMAS_PER_ITER; t++) {
                unsigned dst = buf_addr + t * TILE_BYTES;
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                    " [%0], [%1], %2, [%3], %4;\n"
                    :: "r"(dst), "l"(src + t * (TILE_BYTES / 4)), "r"(TILE_BYTES), "r"(mbar_addr), "h"(mask)
                    : "memory");
            }
        }
        if (tid == 0) {
            int done = 0; int spin = 0;
            while (!done && spin < 1000000) {
                asm volatile("{.reg .pred p;\n"
                             "mbarrier.try_wait.shared.b64 p, [%1], 0;\n"
                             "selp.u32 %0, 1, 0, p;}\n"
                             : "=r"(done) : "r"(mbar_addr) : "memory");
                spin++;
            }
        }
        __syncthreads();
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (tid == 0 && my_cta == 0 && blockIdx.x == 0) {
        out[0] = t1 - t0;
        ((float*)&out[2])[0] = ((float*)buf)[0];
    }
}

int main() {
    CK(cudaSetDevice(0));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    printf("SMs: %d | Cluster size: %d\n\n", sm_count, CX);

    unsigned src_words = 64 * 1024 * 1024;  // 256 MB src in HBM
    float* d_src;
    CK(cudaMalloc(&d_src, src_words * 4));
    cudaMemset(d_src, 0, src_words * 4);

    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    const int N_ITERS = 512;
    CK(cudaFuncSetAttribute(tma_mcast_persistent<N_ITERS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    printf("=== V32 TMA multicast aggregate sweep (%d KB tile, %d iters) ===\n",
           TILE_BYTES/1024, N_ITERS);
    printf("n_clust  CTAs   wall_ms   raw_HBM_TB/s  effective_TB/s   per_iter_us  notes\n");

    for (int n_clusters : {1, 2, 4, 6, 8, 10, 12, 14, 16, 18}) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(n_clusters * CX, 1, 1);
        cfg.blockDim = dim3(128, 1, 1);
        cfg.dynamicSmemBytes = SMEM_BYTES;
        cfg.stream = 0;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = CX;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;

        // Warmup
        cudaError_t e = cudaLaunchKernelEx(&cfg, tma_mcast_persistent<N_ITERS>,
                                           (const float*)d_src, d_out, src_words);
        if (e) {
            printf("%2d       %3d    LAUNCH-FAIL  (%s)\n", n_clusters, n_clusters*CX, cudaGetErrorString(e));
            cudaGetLastError();
            continue;
        }
        e = cudaDeviceSynchronize();
        if (e) {
            printf("%2d       %3d    EXEC-FAIL    (%s)\n", n_clusters, n_clusters*CX, cudaGetErrorString(e));
            cudaGetLastError();
            continue;
        }

        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            cudaLaunchKernelEx(&cfg, tma_mcast_persistent<N_ITERS>,
                              (const float*)d_src, d_out, src_words);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;

        double hbm_bytes = (double)n_clusters * N_ITERS * TILE_BYTES * NUM_TMAS_PER_ITER;
        double delivered_bytes = hbm_bytes * CX;
        double hbm_tbs = hbm_bytes / (avg_ms / 1e3) / 1e12;
        double effective_tbs = delivered_bytes / (avg_ms / 1e3) / 1e12;
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        double per_iter_cy = (double)cy / N_ITERS;
        double per_iter_us = per_iter_cy / 1.920 / 1e3;
        const char* note = "";
        if (hbm_tbs > 7.31 * 1.05) note = " >HBM!";
        if (hbm_tbs < 1.0) note = " low";
        printf("%2d       %3d    %7.3f   %10.3f      %10.3f       %5.2f      %s\n",
               n_clusters, n_clusters*CX, avg_ms, hbm_tbs, effective_tbs, per_iter_us, note);
    }

    printf("\nTheoretical: HBM peak 7.31 TB/s → effective ceiling ~58.4 TB/s (×8 multicast)\n");

    cudaFree(d_src);
    cudaFree(d_out);
    return 0;
}
