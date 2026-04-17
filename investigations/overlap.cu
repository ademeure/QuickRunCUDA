// Test async memcpy + compute overlap on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void compute(float *data, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float a = 0;
    for (int i = 0; i < iters; i++) {
        for (int j = tid; j < N; j += stride) a += data[j] * 1.0001f + 0.0001f;
    }
    if (a < -1e30f) data[tid] = a;
}

int main() {
    cudaSetDevice(0);

    size_t bytes = 64 * 1024 * 1024;
    int N = bytes / sizeof(float);

    float *d_data;
    cudaMalloc(&d_data, bytes);
    void *h_pinned;
    cudaMallocHost(&h_pinned, bytes);
    memset(h_pinned, 0, bytes);

    cudaStream_t s_compute, s_memcpy;
    cudaStreamCreate(&s_compute);
    cudaStreamCreate(&s_memcpy);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    int iters = 10;  // ~3 ms compute

    // Test 1: Memcpy alone
    float t_memcpy = bench([&]{
        cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, s_memcpy);
    });

    // Test 2: Compute alone
    float t_compute = bench([&]{
        compute<<<148, 256, 0, s_compute>>>(d_data, N, iters);
    });

    // Test 3: Sequential (no overlap)
    float t_seq = bench([&]{
        cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, s_compute);
        compute<<<148, 256, 0, s_compute>>>(d_data, N, iters);
    });

    // Test 4: Different streams (potential overlap)
    float t_par = bench([&]{
        cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, s_memcpy);
        compute<<<148, 256, 0, s_compute>>>(d_data, N, iters);
    });

    // Test 5: Forced overlap with event sync
    float t_ev = bench([&]{
        cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, s_memcpy);
        cudaEventRecord(e0, s_memcpy);
        cudaStreamWaitEvent(s_compute, e0, 0);
        compute<<<148, 256, 0, s_compute>>>(d_data, N, iters);
    });

    printf("# B300 async memcpy + compute overlap (64 MB H2D + ~%dms compute)\n\n", iters);
    printf("  memcpy alone:                  %.3f ms\n", t_memcpy);
    printf("  compute alone:                 %.3f ms\n", t_compute);
    printf("  sequential (same stream):      %.3f ms (sum=%.3f)\n", t_seq, t_memcpy+t_compute);
    printf("  parallel (separate streams):   %.3f ms (overlap saving %+.3f)\n",
           t_par, (t_memcpy+t_compute) - t_par);
    printf("  parallel + event dep:          %.3f ms\n", t_ev);
    printf("\n  Overlap efficiency: %.1f%% (vs perfect overlap = %.3f ms)\n",
           ((t_memcpy+t_compute) - t_par) / (std::min(t_memcpy, t_compute)) * 100,
           std::max(t_memcpy, t_compute));

    cudaFree(d_data); cudaFreeHost(h_pinned);
    return 0;
}
