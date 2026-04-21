// R1: FMA pipe gating threshold
// Modulate FMA duty cycle by inserting non-FMA work between FFMAs
// MODE 0..N: ratio of FFMA per other-op
#include <cuda_runtime.h>
#include <cstdio>

__global__ __launch_bounds__(256, 4)
void kernel(float* out, int iters, int u2, int ffma_per_idle) {
    float a = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float b = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float c = 0.5f;
    unsigned int idle_cy = 200;  // ~133 ns per "idle" period

    for (int i = 0; i < iters; i++) {
        // FFMA burst
        for (int k = 0; k < ffma_per_idle; k++) {
            a = a*b + c;
        }
        // Idle period
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < idle_cy);
    }

    if ((int)a == 12345) out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

int main(int argc, char** argv) {
    int ratio = (argc > 1) ? atoi(argv[1]) : 1;
    int iters = 1000000 / (ratio + 1);  // adjust to keep total runtime ~constant
    int blocks = 296;
    int threads = 256;
    float* dev_out;
    cudaMalloc(&dev_out, blocks * threads * sizeof(float));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    kernel<<<blocks, threads>>>(dev_out, 100, 7, ratio);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    kernel<<<blocks, threads>>>(dev_out, iters, 7, ratio);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);

    printf("ratio=%d iters=%d time=%.3f ms (FFMA per idle=200cy)\n", ratio, iters, ms);
    return 0;
}
