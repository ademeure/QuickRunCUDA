// Measure TMA (cp.async.bulk) bandwidth on B300
// Copies from GMEM to SMEM via TMA, using mbarrier for completion
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cuda/barrier>

#define N_ITERS 1000

// Simple TMA bulk copy - GMEM to SMEM
extern "C" __global__ void tma_bulk_test(float *gmem_src, unsigned long long *out) {
    extern __shared__ char smem[];
    __shared__ alignas(8) unsigned long long mbar;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "l"(&mbar));
    }
    __syncthreads();

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    const int bytes = 32768;  // 32 KB per transfer

    for (int i = 0; i < N_ITERS; i++) {
        if (threadIdx.x == 0) {
            // cp.async.bulk.shared::cluster.global
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                "[%0], [%1], %2, [%3];"
                :: "r"((unsigned)__cvta_generic_to_shared(smem)),
                   "l"(gmem_src + (i * 256) % (1024*1024)),
                   "r"(bytes),
                   "r"((unsigned)__cvta_generic_to_shared(&mbar))
                : "memory"
            );
            // Expect-tx for mbarrier
            asm volatile("mbarrier.expect_tx.shared.b64 [%0], %1;"
                         :: "l"(&mbar), "r"(bytes));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            // Wait for mbar
            unsigned long long tok;
            asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
                         : "=l"(tok) : "l"(&mbar));
            while (true) {
                int done;
                asm volatile("{ .reg .pred p; mbarrier.test_wait.shared.b64 p, [%1], %2; selp.s32 %0, 1, 0, p; }"
                             : "=r"(done) : "l"(&mbar), "l"(tok));
                if (done) break;
            }
            // Re-init for next iter
            asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "l"(&mbar));
        }
        __syncthreads();
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (threadIdx.x == 0) out[blockIdx.x] = end - start;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    const int bytes = 32768;
    float *d_src;
    cudaMalloc(&d_src, 4 * 1024 * 1024);  // 16 MB buffer
    cudaMemset(d_src, 0x40, 4 * 1024 * 1024);

    unsigned long long *d_out;
    cudaMalloc(&d_out, sm * sizeof(unsigned long long));

    cudaFuncSetAttribute((void*)tma_bulk_test,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);

    auto t0 = std::chrono::high_resolution_clock::now();
    tma_bulk_test<<<sm, 128, bytes>>>(d_src, d_out);
    cudaDeviceSynchronize();
    cudaError_t r = cudaGetLastError();
    auto t1 = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
    printf("# B300 TMA cp.async.bulk throughput\n");
    printf("# %d blocks × %d bytes × %d iters\n", sm, bytes, N_ITERS);
    printf("# Launch: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));
    printf("# Time: %.4f ms\n", ms);
    if (r == cudaSuccess) {
        unsigned long long h_cycles[256];
        cudaMemcpy(h_cycles, d_out, sm * 8, cudaMemcpyDeviceToHost);
        unsigned long long avg_cy = 0;
        for (int i = 0; i < sm; i++) avg_cy += h_cycles[i];
        avg_cy /= sm;
        printf("# Avg clock64 per kernel: %llu cy\n", avg_cy);
        double ns = avg_cy / 2.032;
        printf("# Per-iter avg: %.1f ns (%llu cy)\n", ns/N_ITERS, avg_cy/N_ITERS);

        size_t total_bytes = (size_t)sm * bytes * N_ITERS;
        printf("# Total bytes: %.2f GB\n", total_bytes/1e9);
        printf("# Aggregate BW: %.1f GB/s\n", total_bytes/(ms/1e3)/1e9);
    }

    cudaFree(d_src); cudaFree(d_out);
    return 0;
}
