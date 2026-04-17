#include <cuda_runtime.h>

__noinline__ __device__ float fma_noinline(float a, float b, float c) {
    return a * b + c;
}

__forceinline__ __device__ float fma_inline(float a, float b, float c) {
    return a * b + c;
}

extern "C" __global__ void noinline_call(float *out, int iters, float k1, float k2) {
    float a = threadIdx.x * 0.001f;
    for (int i = 0; i < iters; i++) {
        a = fma_noinline(a, k1, k2);
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

extern "C" __global__ void inline_call(float *out, int iters, float k1, float k2) {
    float a = threadIdx.x * 0.001f;
    for (int i = 0; i < iters; i++) {
        a = fma_inline(a, k1, k2);
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    float *d; cudaMalloc(&d, 16);
    noinline_call<<<1, 32>>>(d, 100, 1.0f, 0.0f);
    inline_call<<<1, 32>>>(d, 100, 1.0f, 0.0f);
    cudaDeviceSynchronize();
    return 0;
}
