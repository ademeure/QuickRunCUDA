// V33: Per-CTA TMA (no multicast) — push HBM saturation
// Each of N_CTAs independently does cp.async.bulk.shared::cluster.global
// Goal: compare to multicast's 14.9 TB/s effective, see if we can hit HBM peak

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int TILE_BYTES, int N_ITERS>
__global__ __launch_bounds__(128, 1)
void tma_per_cta(const float* src, unsigned long long* out, unsigned total_ctas) {
    extern __shared__ __align__(16) char buf[];
    __shared__ __align__(8) unsigned long long mbar;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
    }
    __syncthreads();

    unsigned buf_addr = (unsigned)__cvta_generic_to_shared(&buf[0]);
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned stride_words = total_ctas * (TILE_BYTES / 4);
    unsigned src_words_mask = (256u * 1024u * 1024u) - (TILE_BYTES / 4);

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        // Stride so each iter reads distinct addresses (defeat L2 caching)
        unsigned off = (bid * (TILE_BYTES / 4) + it * stride_words) & src_words_mask;
        const float* my_src = src + off;

        if (tid == 0) {
            asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbar_addr) : "memory");
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                         :: "r"(mbar_addr), "r"(TILE_BYTES) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %2, [%3];\n"
                :: "r"(buf_addr), "l"(my_src), "r"(TILE_BYTES), "r"(mbar_addr)
                : "memory");
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

    if (tid == 0 && bid == 0) {
        out[0] = t1 - t0;
        ((float*)&out[2])[0] = ((float*)buf)[0];
    }
}

int main() {
    CK(cudaSetDevice(0));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    printf("SMs: %d\n", sm_count);

    unsigned src_words = 256 * 1024 * 1024;  // 1 GB src
    float* d_src;
    CK(cudaMalloc(&d_src, src_words * 4));
    cudaMemset(d_src, 0, src_words * 4);

    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 256;

    // Try several tile sizes and per-CTA counts
    printf("=== V33 per-CTA TMA (no multicast) ===\n");
    printf("tile_KB  n_blocks  wall_ms   raw_HBM_TB/s  per_iter_us\n");

    auto run = [&](int tile_bytes, int n_blocks) {
        const int TILE = tile_bytes;
        int shmem = tile_bytes;
        cudaFuncSetAttribute(tma_per_cta<65536, 256>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_per_cta<32768, 256>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_per_cta<16384, 256>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);

        // warmup
        if (tile_bytes == 16384) tma_per_cta<16384, 256><<<n_blocks, 128, shmem>>>((const float*)d_src, d_out, n_blocks);
        else if (tile_bytes == 32768) tma_per_cta<32768, 256><<<n_blocks, 128, shmem>>>((const float*)d_src, d_out, n_blocks);
        else tma_per_cta<65536, 256><<<n_blocks, 128, shmem>>>((const float*)d_src, d_out, n_blocks);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) {
            printf("%d      %4d      WARMUP-FAIL\n", tile_bytes/1024, n_blocks);
            return;
        }

        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (tile_bytes == 16384) tma_per_cta<16384, 256><<<n_blocks, 128, shmem>>>((const float*)d_src, d_out, n_blocks);
            else if (tile_bytes == 32768) tma_per_cta<32768, 256><<<n_blocks, 128, shmem>>>((const float*)d_src, d_out, n_blocks);
            else tma_per_cta<65536, 256><<<n_blocks, 128, shmem>>>((const float*)d_src, d_out, n_blocks);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;

        double hbm_bytes = (double)n_blocks * N_ITERS * TILE;
        double hbm_tbs = hbm_bytes / (avg_ms / 1e3) / 1e12;
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        double per_iter_us = cy / (double)N_ITERS / 1.920 / 1e3;
        printf("%d      %4d      %7.3f   %10.3f    %5.2f\n",
               tile_bytes/1024, n_blocks, avg_ms, hbm_tbs, per_iter_us);
    };

    for (int blocks : {16, 32, 64, 128, 148}) {
        run(16384, blocks);
    }
    for (int blocks : {16, 32, 64, 128, 148}) {
        run(32768, blocks);
    }
    for (int blocks : {16, 32, 64, 128, 148}) {
        run(65536, blocks);
    }

    printf("\nHBM peak 7.31 TB/s (A6 rigor).\n");

    cudaFree(d_src);
    cudaFree(d_out);
    return 0;
}
