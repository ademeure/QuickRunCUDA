// G2: -use_fast_math impact on speed and power
// Same FFMA + transcendental kernel, with vs without fast_math
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__global__ __launch_bounds__(256, 4)
void kernel(float* out, int iters, int u2) {
    float a = (float)threadIdx.x * 0.01f + (float)u2 * 1e-9f;
    float b = (float)(threadIdx.x ^ u2) * 0.02f + 1.0f;

    for (int i = 0; i < iters; i++) {
        // Mix of FFMA + transcendental (rsqrt, exp, log) to expose fast_math differences
        a = a * b + 0.5f;
        a = rsqrtf(a + 1.0f);
        a = expf(a);
        a = logf(a + 1.0f);
        a = a * b + 0.5f;
        a = a * b + 0.5f;
        a = a * b + 0.5f;
        a = sinf(a);
        a = cosf(a);
        a = a * b + 0.5f;
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

    // Warmup
    kernel<<<blocks, threads>>>(dev_out, 1000, 7);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    kernel<<<blocks, threads>>>(dev_out, iters, 7);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);

#ifdef FAST_MATH
    printf("FAST_MATH=ON  time=%.3f ms\n", ms);
#else
    printf("FAST_MATH=OFF time=%.3f ms\n", ms);
#endif

    cudaFree(dev_out);
    return 0;
}
