// V5 D2: per-pipe duty cycle vs power
// Vary FFMA-vs-IADD3 ratio; measure power
// FFMA hot pipe; IADD3 lighter; mix scales between
#include <cuda_runtime.h>
#include <cstdio>

#ifndef DUTY
#define DUTY 16  // FFMA per IADD3
#endif

__global__ __launch_bounds__(256, 4)
void kernel(float* out, int iters, int u2) {
    float a = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float b = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float c = 0.5f;
    unsigned int v = (unsigned)u2;

    for (int i = 0; i < iters; i++) {
#pragma unroll 16
        for (int u = 0; u < 16; u++) {
            // DUTY FFMAs per 1 IADD3
            #pragma unroll
            for (int k = 0; k < DUTY; k++) a = a*b + c;
            v += i;
        }
    }
    if ((int)a == 12345 && v == 0xDEADBEEF) out[blockIdx.x * blockDim.x + threadIdx.x] = a + (float)v;
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 200000;
    int blocks = 296, threads = 256;
    float* d;
    cudaMalloc(&d, blocks * threads * sizeof(float));
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    kernel<<<blocks, threads>>>(d, 1000, 7);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    kernel<<<blocks, threads>>>(d, iters, 7);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    printf("DUTY=%d time=%.3f ms\n", DUTY, ms);
    return 0;
}
