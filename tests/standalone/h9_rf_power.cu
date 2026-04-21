// H9: Register file access power
// Compare FFMA with .reuse (saves RF reads) vs without
// Same logical work but different RF read counts
#include <cuda_runtime.h>
#include <cstdio>

#ifndef MODE
#define MODE 0
#endif

__global__ __launch_bounds__(256, 4)
void kernel(float* out, int iters, int u2) {
    // 16 indep chains
    float a[16], ya[16];
    float za = 0.5f + (float)u2 * 1e-9f;
    for (int k = 0; k < 16; k++) {
        a[k] = (float)(threadIdx.x ^ u2) * 0.001f * (k+1);
        ya[k] = (float)(threadIdx.x ^ (u2+k)) * 0.002f + 1.0f;
    }

    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // Broadcast za → compiler emits .reuse on za (1 RF read effective)
            for (int k = 0; k < 16; k++) {
                asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a[k]) : "f"(ya[k]), "f"(za));
            }
#else
            // Per-chain za → no .reuse (3 RF reads per FMA)
            // Use distinct values
            for (int k = 0; k < 16; k++) {
                float zak = za + (float)(k * 0.001f);
                asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a[k]) : "f"(ya[k]), "f"(zak));
            }
#endif
        }
    }

    float sink = 0;
    for (int k = 0; k < 16; k++) sink += a[k];
    if ((int)sink == 12345) out[blockIdx.x * blockDim.x + threadIdx.x] = sink;
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 100000;
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

#ifdef MODE0
    printf("MODE=0 (broadcast .reuse) time=%.3f ms\n", ms);
#else
    printf("MODE=%d time=%.3f ms\n", MODE, ms);
#endif

    return 0;
}
