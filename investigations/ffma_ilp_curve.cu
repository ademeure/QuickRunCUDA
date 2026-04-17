// Maximum FFMA throughput with high ILP
#include <cuda_runtime.h>
#include <cstdio>

template<int ILP>
__global__ void ffma_max(float *out, int iters, float k1, float k2) {
    float a[ILP];
    for (int i = 0; i < ILP; i++) a[i] = threadIdx.x * (i + 1) * 0.001f;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) a[j] = a[j] * k1 + k2;
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < ILP; j++) s += a[j];
    if (s < -1e30f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 148 * 256 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;
    auto bench = [&](auto launch, int ilp) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total_flops = (long)blocks * threads * iters * ilp * 2;
        double tflops = total_flops / (best/1000.0) / 1e12;
        printf("  ILP=%d: %.3f ms = %.1f TFLOPS (%.0f%% of 76.96 theoretical peak)\n",
               ilp, best, tflops, tflops/76.96*100);
    };

    printf("# B300 FFMA peak vs ILP (148 blocks × 128 threads, 100k iter)\n\n");
    bench([&]{ ffma_max<1><<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f); }, 1);
    bench([&]{ ffma_max<2><<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f); }, 2);
    bench([&]{ ffma_max<4><<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f); }, 4);
    bench([&]{ ffma_max<8><<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f); }, 8);
    bench([&]{ ffma_max<16><<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f); }, 16);
    bench([&]{ ffma_max<32><<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f); }, 32);

    return 0;
}
