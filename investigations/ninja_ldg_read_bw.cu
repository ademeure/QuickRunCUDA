// Establish LDG.E.128 raw read BW for cross-check vs HBM 7.31 TB/s
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 4) __global__ void ldg_read(const uint4 *__restrict__ gmem, uint4 *sink, int N_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0,0,0,0);
    for (int i = 0; i < N_iters; i++) {
        uint4 v = gmem[tid + i * stride];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0xdeadbeef) sink[0] = acc;
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;
    void *d_data, *d_sink;
    cudaMalloc(&d_data, bytes); cudaMemset(d_data, 0xab, bytes);
    cudaMalloc(&d_sink, 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    
    // Sweep blocks to find optimum
    for (int blocks : {148, 296, 592, 1184, 2368, 4736, 9472, 18944, 37888}) {
        int threads = 256;
        size_t per_kernel = bytes;
        int N_iters = per_kernel / ((size_t)blocks * threads * 16);
        for (int i = 0; i < 3; i++) ldg_read<<<blocks, threads>>>((uint4*)d_data, (uint4*)d_sink, N_iters);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 20; i++) {
            cudaEventRecord(e0);
            ldg_read<<<blocks, threads>>>((uint4*)d_data, (uint4*)d_sink, N_iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        size_t total = (size_t)blocks * threads * N_iters * 16;
        double gbs = total / (best/1000.0) / 1e9;
        printf("  blocks=%5d iters=%5d total=%5.1f GB %.4f ms = %.0f GB/s = %.2f%% spec\n",
               blocks, N_iters, total/1e9, best, gbs, gbs/7672*100);
    }
    return 0;
}
