// H10: Predicate RF power — heavy setp activity
// MODE 0: pure FFMA baseline
// MODE 1: FFMA + setp every iter (predicate set but unused)
// MODE 2: FFMA + selp every iter (predicate read)
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
    unsigned int v = (unsigned)(threadIdx.x ^ u2);

    for (int i = 0; i < iters; i++) {
#pragma unroll 32
        for (int u = 0; u < 32; u++) {
#if MODE == 0
            // Pure FFMA baseline
            a = a*b + c;
#elif MODE == 1
            // FFMA + setp (predicate set every iter, never used)
            a = a*b + c;
            asm volatile("{ .reg .pred p; setp.ne.b32 p, %0, 0; }" :: "r"(v));
#elif MODE == 2
            // FFMA + setp + selp (predicate read used)
            asm volatile("{ .reg .pred p; setp.ne.b32 p, %0, 0;\n"
                         "  selp.b32 %0, %1, 0, p; }"
                         : "+r"(v) : "r"(v + (unsigned)i));
            a = a*b + c;
#endif
        }
    }

    if ((int)a == 12345 && v == 0xDEADBEEF) out[blockIdx.x * blockDim.x + threadIdx.x] = a + (float)v;
}

int main(int argc, char** argv) {
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
    printf("MODE=%d time=%.3f ms\n", MODE, ms);
    return 0;
}
