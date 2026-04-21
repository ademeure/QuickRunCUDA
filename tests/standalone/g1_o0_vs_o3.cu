// G1: -O0 vs -O3 SASS divergence
// Same kernel; observe SASS instruction count + runtime
#include <cuda_runtime.h>
#include <cstdio>

__global__ __launch_bounds__(256, 4)
void kernel(float* out, int iters, int u2) {
    float a = (float)threadIdx.x * 0.01f + (float)u2 * 1e-9f;
    float b = (float)(threadIdx.x ^ u2) * 0.02f + 1.0f;
    float c = 0.5f;

    for (int i = 0; i < iters; i++) {
        a = a * b + c;
        a = a * b + c;
        a = a * b + c;
        a = a * b + c;
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

#ifdef OPT_LEVEL
    printf("OPT=%s time=%.3f ms\n", OPT_LEVEL, ms);
#else
    printf("OPT=unknown time=%.3f ms\n", ms);
#endif

    cudaFree(dev_out);
    return 0;
}
