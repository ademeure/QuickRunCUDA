// NINJA TMA: try cp.async.bulk for HBM write to see if TMA beats 98.7% v8 store
// TMA can transfer larger chunks per instruction; lower instruction overhead.
#include <cuda_runtime.h>
#include <cstdio>

#ifndef SIZE_KB
#define SIZE_KB 4
#endif

// Need __cluster_dims__ for TMA, plus mbarrier setup
extern "C" __launch_bounds__(128, 1) __global__ void w_tma_bulk(int *data, int N_iters) {
    extern __shared__ int smem[];
    int t = threadIdx.x;
    if (t < SIZE_KB * 256) smem[t] = 0xab;  // initialize SMEM
    __syncthreads();

    // bulk store smem -> global
    int *gdst = data + blockIdx.x * (SIZE_KB * 256);
    if (t == 0) {
        for (int i = 0; i < N_iters; i++) {
            int *dst = gdst + i * gridDim.x * (SIZE_KB * 256);
            asm volatile(
                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                :: "l"(dst), "l"(smem), "n"(SIZE_KB * 1024)
                : "memory");
        }
        asm volatile("cp.async.bulk.commit_group;");
        asm volatile("cp.async.bulk.wait_group 0;");
    }
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;
    int *d; cudaMalloc(&d, bytes);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int blocks = 256;
    int N_iters = bytes / (SIZE_KB * 1024 * blocks);

    for (int i = 0; i < 3; i++)
        w_tma_bulk<<<blocks, 128, SIZE_KB * 1024>>>(d, N_iters);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ERR: %s\n", cudaGetErrorString(err)); return 1; }

    float best = 1e30f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(e0);
        w_tma_bulk<<<blocks, 128, SIZE_KB * 1024>>>(d, N_iters);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    double gbs = bytes / (best/1000) / 1e9;
    printf("# TMA bulk SIZE_KB=%d, blocks=%d, iters=%d: %.4f ms = %.1f GB/s = %.2f%% spec\n",
           SIZE_KB, blocks, N_iters, best, gbs, gbs/7672*100);
    return 0;
}
