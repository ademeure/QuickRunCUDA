// Measure SHMEM peak bandwidth properly (avoid loop/compiler hoist artifacts)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

// Many independent loads per iter — load coalesced 32 lanes at once
// Each warp issues `READS_PER_ITER` LDS instructions per loop step
// Use registers as accumulators, write back at end
extern "C" __global__ void shmem_peak_warp(float *out, int n_iters) {
    extern __shared__ float buf[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    // Initialize a chunk for each warp
    for (int i = tid; i < 4096; i += blockDim.x) buf[i] = (float)i;
    __syncthreads();

    // 8 register accumulators per thread, each loads from a different bank
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;

    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        // Coalesced reads: each thread reads `lane` + i*32 from each bank stripe
        int base = (i * 256) & (4096 - 256);
        a0 += buf[base + 0   + lane];
        a1 += buf[base + 32  + lane];
        a2 += buf[base + 64  + lane];
        a3 += buf[base + 96  + lane];
        a4 += buf[base + 128 + lane];
        a5 += buf[base + 160 + lane];
        a6 += buf[base + 192 + lane];
        a7 += buf[base + 224 + lane];
    }

    // Defeat DCE
    if (a0+a1+a2+a3+a4+a5+a6+a7 < -1e30f) out[blockIdx.x] = a0;
    if (tid == 0) out[blockIdx.x] = a0;  // unconditional write
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    float *d_out;
    cudaMalloc(&d_out, sm * 32 * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);
    cudaFuncSetAttribute((void*)shmem_peak_warp,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    int n_iters = 1000;
    int threads = 256;
    int blocks = sm;

    float t = bench([&]{
        shmem_peak_warp<<<blocks, threads, 16384, s>>>(d_out, n_iters);
    });

    // Each thread does READS_PER_ITER * n_iters loads, each 4 bytes
    int reads_per_iter = 8;
    long long total_bytes = (long long)blocks * threads * n_iters * reads_per_iter * 4;
    float bw_aggr = total_bytes / (t/1e3) / 1e9;
    float bw_per_sm = bw_aggr / sm;

    printf("# B300 SHMEM peak BW (proper test)\n");
    printf("# %d blocks × %d threads, %d reads_per_iter × %d iter\n",
           blocks, threads, reads_per_iter, n_iters);
    printf("# Time: %.4f ms\n", t);
    printf("# Bytes: %lld (~%.2f GB)\n", total_bytes, total_bytes/1e9);
    printf("# Aggregate BW: %.1f GB/s (%.2f TB/s)\n", bw_aggr, bw_aggr/1000);
    printf("# Per-SM BW: %.1f GB/s\n", bw_per_sm);
    printf("# Theoretical peak ≈ 32 banks × 4 bytes/cy × 2.032 GHz × 148 SMs = %.1f TB/s\n",
           32.0 * 4 * 2.032 * sm / 1000);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
