// V34: TMA WRITE peak (cp.async.bulk.global.shared::cluster)
// Theoretical: HBM write peak ~6.7 TB/s (measured A2 7.07 TB/s at unbalanced cases)
// Apply 10-rule rigor.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int TILE_BYTES, int N_ITERS>
__global__ __launch_bounds__(128, 1)
void tma_write(float* dst, unsigned long long* out, unsigned total_ctas) {
    extern __shared__ __align__(16) float buf[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // Fill smem with non-zero data so writes are meaningful
    for (int i = tid; i < TILE_BYTES/4; i += 128) {
        buf[i] = (float)(i ^ bid);
    }
    __syncthreads();

    unsigned buf_addr = (unsigned)__cvta_generic_to_shared(&buf[0]);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    size_t stride_words = total_ctas * (TILE_BYTES / 4);
    size_t dst_cap = 1ull * 1024 * 1024 * 1024;

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        size_t off = (bid * (size_t)(TILE_BYTES / 4) + (size_t)it * stride_words);
        off %= (dst_cap - TILE_BYTES / 4);
        float* my_dst = dst + off;

        if (tid == 0) {
            asm volatile(
                "cp.async.bulk.global.shared::cta.bulk_group"
                " [%0], [%1], %2;\n"
                :: "l"(my_dst), "r"(buf_addr), "r"(TILE_BYTES)
                : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
        __syncthreads();
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0 && bid == 0) {
        out[0] = t1 - t0;
        ((float*)&out[2])[0] = buf[0];
    }
}

int main() {
    CK(cudaSetDevice(0));

    size_t dst_words = 1ull * 1024 * 1024 * 1024;  // 4 GB dst
    float* d_dst;
    CK(cudaMalloc(&d_dst, dst_words * 4));

    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 512;

    printf("=== V34 TMA WRITE (cp.async.bulk.global.shared::cluster) ===\n");
    printf("tile_KB  n_blocks  wall_ms   raw_HBM_TB/s  per_iter_us\n");

    auto run16 = [&](int n_blocks) {
        const int TILE = 16384;
        cudaFuncSetAttribute(tma_write<16384, 512>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        tma_write<16384, 512><<<n_blocks, 128, TILE>>>(d_dst, d_out, n_blocks);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) {
            printf("%d      %4d      FAIL\n", TILE/1024, n_blocks);
            return;
        }
        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            tma_write<16384, 512><<<n_blocks, 128, TILE>>>(d_dst, d_out, n_blocks);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double hbm_bytes = (double)n_blocks * N_ITERS * TILE;
        double hbm_tbs = hbm_bytes / (avg_ms / 1e3) / 1e12;
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        double per_iter_us = cy / (double)N_ITERS / 1.920 / 1e3;
        printf("%d      %4d      %7.3f   %10.3f    %5.2f\n", TILE/1024, n_blocks, avg_ms, hbm_tbs, per_iter_us);
    };
    auto run32 = [&](int n_blocks) {
        const int TILE = 32768;
        cudaFuncSetAttribute(tma_write<32768, 512>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        tma_write<32768, 512><<<n_blocks, 128, TILE>>>(d_dst, d_out, n_blocks);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("%d      %4d      FAIL\n", TILE/1024, n_blocks); return; }
        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            tma_write<32768, 512><<<n_blocks, 128, TILE>>>(d_dst, d_out, n_blocks);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double hbm_bytes = (double)n_blocks * N_ITERS * TILE;
        double hbm_tbs = hbm_bytes / (avg_ms / 1e3) / 1e12;
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        double per_iter_us = cy / (double)N_ITERS / 1.920 / 1e3;
        printf("%d      %4d      %7.3f   %10.3f    %5.2f\n", TILE/1024, n_blocks, avg_ms, hbm_tbs, per_iter_us);
    };
    auto run64 = [&](int n_blocks) {
        const int TILE = 65536;
        cudaFuncSetAttribute(tma_write<65536, 512>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        tma_write<65536, 512><<<n_blocks, 128, TILE>>>(d_dst, d_out, n_blocks);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("%d      %4d      FAIL\n", TILE/1024, n_blocks); return; }
        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            tma_write<65536, 512><<<n_blocks, 128, TILE>>>(d_dst, d_out, n_blocks);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double hbm_bytes = (double)n_blocks * N_ITERS * TILE;
        double hbm_tbs = hbm_bytes / (avg_ms / 1e3) / 1e12;
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        double per_iter_us = cy / (double)N_ITERS / 1.920 / 1e3;
        printf("%d      %4d      %7.3f   %10.3f    %5.2f\n", TILE/1024, n_blocks, avg_ms, hbm_tbs, per_iter_us);
    };

    for (int b : {64, 128, 148}) run16(b);
    for (int b : {64, 128, 148}) run32(b);
    for (int b : {64, 128, 148}) run64(b);

    printf("\nHBM write peak ~6.7 TB/s (A2/A6 rigor) — looking for >90%%\n");

    cudaFree(d_dst);
    cudaFree(d_out);
    return 0;
}
