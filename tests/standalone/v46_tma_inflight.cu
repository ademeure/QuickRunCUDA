// V46: TMA in-flight depth — how many TMAs can a single CTA queue up?
// Issue N TMAs back-to-back into separate buffers, then wait for all.
// Throughput vs N tells us the queue depth.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int TILE_BYTES, int N_INFLIGHT, int N_ITERS>
__global__ __launch_bounds__(128, 1)
void tma_inflight(const float* src, unsigned long long* out, unsigned total_ctas) {
    extern __shared__ __align__(16) char buf_raw[];
    __shared__ __align__(8) unsigned long long mbar[16];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < N_INFLIGHT; i++) {
            asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                         :: "r"((unsigned)__cvta_generic_to_shared(&mbar[i])) : "memory");
        }
    }
    __syncthreads();

    unsigned bufs[16];
    unsigned mbars[16];
    #pragma unroll
    for (int i = 0; i < N_INFLIGHT; i++) {
        bufs[i] = (unsigned)__cvta_generic_to_shared(&buf_raw[i * TILE_BYTES]);
        mbars[i] = (unsigned)__cvta_generic_to_shared(&mbar[i]);
    }

    size_t stride = total_ctas * (TILE_BYTES / 4);
    size_t cap = 1ull * 1024 * 1024 * 1024;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        if (tid == 0) {
            #pragma unroll
            for (int i = 0; i < N_INFLIGHT; i++) {
                asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbars[i]) : "memory");
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                             :: "r"(mbars[i]), "r"(TILE_BYTES) : "memory");
                size_t off = (bid * (size_t)(TILE_BYTES / 4) + (size_t)(it * N_INFLIGHT + i) * stride) % (cap - TILE_BYTES/4);
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1], %2, [%3];\n"
                             :: "r"(bufs[i]), "l"(src + off), "r"(TILE_BYTES), "r"(mbars[i])
                             : "memory");
            }
            // Wait for all
            #pragma unroll
            for (int i = 0; i < N_INFLIGHT; i++) {
                int done = 0; int spin = 0;
                while (!done && spin < 1000000) {
                    asm volatile("{.reg .pred p;\n"
                                 "mbarrier.try_wait.shared.b64 p, [%1], 0;\n"
                                 "selp.u32 %0, 1, 0, p;}\n"
                                 : "=r"(done) : "r"(mbars[i]) : "memory");
                    spin++;
                }
            }
        }
        __syncthreads();
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0 && bid == 0) {
        out[0] = t1 - t0;
        ((float*)&out[2])[0] = ((float*)buf_raw)[0];
    }
}

int main() {
    CK(cudaSetDevice(0));

    size_t words = 1ull * 1024 * 1024 * 1024;
    float* d_src;
    CK(cudaMalloc(&d_src, words * 4));
    cudaMemset(d_src, 0, words * 4);

    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 256;
    const int TILE = 16384;  // 16 KB per TMA

    printf("=== V46 TMA in-flight depth (148 blocks × %d KB/TMA × %d iters) ===\n", TILE/1024, N_ITERS);
    printf("Total bytes per iter = N_INFLIGHT × tile per CTA\n\n");
    printf("N_inflight  shmem_KB  wall_ms   HBM_TB/s   per_iter_us\n");

    auto run = [&](int n_inflight) {
        int shmem = n_inflight * TILE;
        if (shmem > 163840) { printf("%2d         too big\n", n_inflight); return; }

        if (n_inflight == 1) {
            cudaFuncSetAttribute(tma_inflight<TILE, 1, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
            tma_inflight<TILE, 1, 256><<<148, 128, shmem>>>((const float*)d_src, d_out, 148);
        } else if (n_inflight == 2) {
            cudaFuncSetAttribute(tma_inflight<TILE, 2, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
            tma_inflight<TILE, 2, 256><<<148, 128, shmem>>>((const float*)d_src, d_out, 148);
        } else if (n_inflight == 4) {
            cudaFuncSetAttribute(tma_inflight<TILE, 4, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
            tma_inflight<TILE, 4, 256><<<148, 128, shmem>>>((const float*)d_src, d_out, 148);
        } else if (n_inflight == 8) {
            cudaFuncSetAttribute(tma_inflight<TILE, 8, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
            tma_inflight<TILE, 8, 256><<<148, 128, shmem>>>((const float*)d_src, d_out, 148);
        }
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("%2d FAIL\n", n_inflight); return; }

        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (n_inflight == 1) tma_inflight<TILE, 1, 256><<<148, 128, shmem>>>((const float*)d_src, d_out, 148);
            else if (n_inflight == 2) tma_inflight<TILE, 2, 256><<<148, 128, shmem>>>((const float*)d_src, d_out, 148);
            else if (n_inflight == 4) tma_inflight<TILE, 4, 256><<<148, 128, shmem>>>((const float*)d_src, d_out, 148);
            else if (n_inflight == 8) tma_inflight<TILE, 8, 256><<<148, 128, shmem>>>((const float*)d_src, d_out, 148);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double bytes = (double)148 * N_ITERS * n_inflight * TILE;
        double tbs = bytes / (avg_ms / 1e3) / 1e12;
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        double per_iter_us = cy / (double)N_ITERS / 1.920 / 1e3;
        printf("%2d         %5d     %7.3f   %8.3f    %5.2f\n", n_inflight, shmem/1024, avg_ms, tbs, per_iter_us);
    };

    for (int n : {1, 2, 4, 8}) run(n);

    printf("\nIf single-CTA TMA is queue-limited at depth N, larger N → higher BW per iter.\n");

    cudaFree(d_src);
    cudaFree(d_out);
    return 0;
}
