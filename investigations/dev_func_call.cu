// __device__ function call overhead: inline vs noinline
#include <cuda_runtime.h>
#include <cstdio>

__forceinline__ __device__ float fma_inline(float a, float b, float c) {
    return a * b + c;
}

__noinline__ __device__ float fma_noinline(float a, float b, float c) {
    return a * b + c;
}

extern "C" __global__ void inline_call(float *out, int iters, float k1, float k2) {
    float a = threadIdx.x * 0.001f;
    for (int i = 0; i < iters; i++) {
        a = fma_inline(a, k1, k2);
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

extern "C" __global__ void noinline_call(float *out, int iters, float k1, float k2) {
    float a = threadIdx.x * 0.001f;
    for (int i = 0; i < iters; i++) {
        a = fma_noinline(a, k1, k2);
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

extern "C" __global__ void no_call(float *out, int iters, float k1, float k2) {
    float a = threadIdx.x * 0.001f;
    for (int i = 0; i < iters; i++) {
        a = a * k1 + k2;
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;

    auto bench = [&](auto launch, const char *name) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        printf("  %-30s %.3f ms\n", name, best);
    };

    printf("# B300 device function call overhead\n");
    printf("# 148 × 128 thr × 100k iter of FMA\n\n");
    bench([&]{ no_call<<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f); }, "no function (inline manually)");
    bench([&]{ inline_call<<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f); }, "__forceinline__ call");
    bench([&]{ noinline_call<<<blocks, threads>>>(d_out, iters, 1.0001f, 0.0001f); }, "__noinline__ call");

    return 0;
}
