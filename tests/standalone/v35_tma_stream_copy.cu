// V35: TMA stream copy — combined read+write HBM via SMEM pipeline
// Theoretical: A6 showed 50:50 R:W @ LDG+STG = 6.68 TB/s combined.
// Can TMA stream copy match? Ceiling for cluster-local copies?

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int TILE_BYTES, int N_ITERS>
__global__ __launch_bounds__(128, 1)
void tma_copy(const float* src, float* dst, unsigned long long* out, unsigned total_ctas) {
    extern __shared__ __align__(16) char buf[];
    __shared__ __align__(8) unsigned long long mbar;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid == 0) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
    __syncthreads();

    unsigned buf_addr = (unsigned)__cvta_generic_to_shared(&buf[0]);
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    size_t stride = total_ctas * (TILE_BYTES / 4);
    size_t cap = 1ull * 1024 * 1024 * 1024;

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        size_t off = (bid * (size_t)(TILE_BYTES / 4) + (size_t)it * stride) % (cap - TILE_BYTES/4);
        const float* my_src = src + off;
        float* my_dst = dst + off;

        // READ from global into smem
        if (tid == 0) {
            asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbar_addr) : "memory");
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                         :: "r"(mbar_addr), "r"(TILE_BYTES) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
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

        // WRITE from smem to global
        if (tid == 0) {
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
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
        ((float*)&out[2])[0] = ((float*)buf)[0];
    }
}

int main() {
    CK(cudaSetDevice(0));

    size_t words = 1ull * 1024 * 1024 * 1024;  // 4 GB each
    float *d_src, *d_dst;
    CK(cudaMalloc(&d_src, words * 4));
    CK(cudaMalloc(&d_dst, words * 4));
    cudaMemset(d_src, 0, words * 4);
    cudaMemset(d_dst, 0, words * 4);

    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 512;

    printf("=== V35 TMA stream COPY (read + write via smem pipeline) ===\n");
    printf("tile_KB  n_blocks  wall_ms   R+W_bytes_TB/s  per_iter_us\n");

    auto run = [&](int tile, int blocks, int shmem) {
        cudaFuncSetAttribute(tma_copy<16384, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_copy<32768, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_copy<65536, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);

        if (tile == 16384) tma_copy<16384, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
        else if (tile == 32768) tma_copy<32768, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
        else tma_copy<65536, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("%d      %4d      FAIL\n", tile/1024, blocks); return; }

        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (tile == 16384) tma_copy<16384, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
            else if (tile == 32768) tma_copy<32768, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
            else tma_copy<65536, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double rw_bytes = (double)blocks * N_ITERS * tile * 2;  // read + write
        double rw_tbs = rw_bytes / (avg_ms / 1e3) / 1e12;
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        double per_iter_us = cy / (double)N_ITERS / 1.920 / 1e3;
        printf("%d      %4d      %7.3f   %12.3f      %5.2f\n", tile/1024, blocks, avg_ms, rw_tbs, per_iter_us);
    };

    for (int b : {64, 128, 148}) run(16384, b, 16384);
    for (int b : {64, 128, 148}) run(32768, b, 32768);
    for (int b : {64, 128, 148}) run(65536, b, 65536);

    printf("\nHBM R+W peak (A6 50:50): 6.68 TB/s. Pure R or W: 7.31 TB/s.\n");

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_out);
    return 0;
}
