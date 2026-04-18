// Sanity: pure copy at same access pattern as RMS-norm (1 row per warp)
#include <cuda_runtime.h>
#include <cstdio>
constexpr int D = 4096;
constexpr int N_ROWS = 1024 * 1024;

__launch_bounds__(256, 8) __global__ void k_copy(const uint4 *x, uint4 *y) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    long row = (long)blockIdx.x * 8 + warp_id;
    if (row >= N_ROWS) return;
    constexpr int D8 = D / 8;  // 512 uint4 per row
    constexpr int PER_THREAD = D8 / 32;  // 16
    #pragma unroll
    for (int k = 0; k < PER_THREAD; k++) {
        uint4 v = x[row * D8 + k * 32 + lane];
        y[row * D8 + k * 32 + lane] = v;
    }
}

int main() {
    cudaSetDevice(0);
    size_t bytes = (size_t)N_ROWS * D * 2;
    void *d_x, *d_y; cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    cudaMemset(d_x, 0x3c, bytes);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = (N_ROWS + 7) / 8;
    auto launch = [&]() { k_copy<<<blocks, 256>>>((uint4*)d_x, (uint4*)d_y); };
    for (int i = 0; i < 3; i++) launch();
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 8; i++) {
        cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    double tbs = (double)(2 * bytes) / (best/1000.0) / 1e12;
    printf("# Pure copy at warp-per-row pattern: %.3f ms = %.2f TB/s\n", best, tbs);
    return 0;
}
