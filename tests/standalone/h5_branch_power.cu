// H5: branch pipe power — BRA-heavy kernel vs FFMA baseline
#include <cuda_runtime.h>
#include <cstdio>

#ifndef MODE
#define MODE 0
#endif

__global__ __launch_bounds__(256, 4)
void kernel(float* out, int iters, int u2) {
    float a = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float b = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float c = 0.5f;
    int v = threadIdx.x ^ u2;

    for (int i = 0; i < iters; i++) {
#pragma unroll 32
        for (int u = 0; u < 32; u++) {
#if MODE == 0
            // FFMA baseline
            a = a*b + c;
#elif MODE == 1
            // BRA-heavy: every iteration takes a (predictable) branch
            if (v + i != 0xFFFFFFFF) {
                a = a*b + c;
            } else {
                a = a*b - c;
            }
#elif MODE == 2
            // BRA + divergent: half-warp branch
            if ((threadIdx.x & 1) ^ ((v + i) & 1)) {
                a = a*b + c;
            } else {
                a = a*b - c;
            }
#endif
            v ^= i;
        }
    }

    if ((int)a == 12345 && v == 0) out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 200000;
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
    printf("MODE=%d time=%.3f ms\n", MODE, ms);

    return 0;
}
