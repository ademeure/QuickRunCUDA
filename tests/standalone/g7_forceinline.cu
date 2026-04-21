// G7: __forceinline__ vs no-inline (rely on default)
// Same kernel calling helper many times; with vs without __forceinline__
#include <cuda_runtime.h>
#include <cstdio>

#ifndef FORCE_INLINE
#define FORCE_INLINE 0
#endif

#if FORCE_INLINE
#define INLINE __device__ __forceinline__
#else
#define INLINE __device__ __noinline__
#endif

INLINE float helper(float a, float b, float c) {
    return a * b + c;
}

__global__ __launch_bounds__(256, 4)
void kernel(float* out, int iters, int u2) {
    float a = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float b = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float c = 0.5f;

    for (int i = 0; i < iters; i++) {
        a = helper(a, b, c);
        a = helper(a, b, c);
        a = helper(a, b, c);
        a = helper(a, b, c);
    }

    if ((int)a == 12345) out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

int main() {
    int iters = 1000000;
    int blocks = 296;
    int threads = 256;
    float* dev_out;
    cudaMalloc(&dev_out, blocks * threads * sizeof(float));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    kernel<<<blocks, threads>>>(dev_out, 1000, 7);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    kernel<<<blocks, threads>>>(dev_out, iters, 7);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);

#if FORCE_INLINE
    printf("FORCEINLINE=1 time=%.3f ms\n", ms);
#else
    printf("NOINLINE time=%.3f ms\n", ms);
#endif

    return 0;
}
