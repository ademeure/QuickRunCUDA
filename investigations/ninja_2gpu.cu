// NINJA: 2-GPU concurrent read peak — does each GPU sustain its own ~7.3?
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void r_only(const int *data, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    const int *base = data + warp_id * (2 * 32 * 8);
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < 2; it++) {
        const int *p = base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(p));
        acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
    else if (tid == 0) out[0] = acc;
}

int main() {
    int n; cudaGetDeviceCount(&n);
    if (n < 2) { printf("Need 2 GPUs\n"); return 1; }

    size_t bytes = 4ull * 1024 * 1024 * 1024;
    int *d0, *d1;
    cudaSetDevice(0); cudaMalloc(&d0, bytes); cudaMemset(d0, 0xab, bytes);
    int *o0; cudaMalloc(&o0, 1<<24);
    cudaSetDevice(1); cudaMalloc(&d1, bytes); cudaMemset(d1, 0xab, bytes);
    int *o1; cudaMalloc(&o1, 1<<24);

    cudaSetDevice(0);
    cudaStream_t s0, s1;
    cudaStreamCreate(&s0);
    cudaSetDevice(1);
    cudaStreamCreate(&s1);

    cudaSetDevice(0);
    cudaEvent_t e0a, e0b; cudaEventCreate(&e0a); cudaEventCreate(&e0b);
    cudaSetDevice(1);
    cudaEvent_t e1a, e1b; cudaEventCreate(&e1a); cudaEventCreate(&e1b);

    int blocks = bytes / (256 * 64);

    // Warmup both
    cudaSetDevice(0);
    for (int i = 0; i < 5; i++) r_only<<<blocks, 256, 0, s0>>>(d0, o0);
    cudaSetDevice(1);
    for (int i = 0; i < 5; i++) r_only<<<blocks, 256, 0, s1>>>(d1, o1);
    cudaDeviceSynchronize();
    cudaSetDevice(0); cudaDeviceSynchronize();

    // Concurrent run
    float best0 = 1e30f, best1 = 1e30f;
    for (int rep = 0; rep < 10; rep++) {
        cudaSetDevice(0);
        cudaEventRecord(e0a, s0);
        r_only<<<blocks, 256, 0, s0>>>(d0, o0);
        cudaEventRecord(e0b, s0);
        cudaSetDevice(1);
        cudaEventRecord(e1a, s1);
        r_only<<<blocks, 256, 0, s1>>>(d1, o1);
        cudaEventRecord(e1b, s1);

        cudaEventSynchronize(e0b);
        cudaEventSynchronize(e1b);
        float ms0, ms1;
        cudaEventElapsedTime(&ms0, e0a, e0b);
        cudaEventElapsedTime(&ms1, e1a, e1b);
        if (ms0 < best0) best0 = ms0;
        if (ms1 < best1) best1 = ms1;
    }

    double g0 = bytes / (best0/1000) / 1e9;
    double g1 = bytes / (best1/1000) / 1e9;
    printf("# 2-GPU concurrent read:\n");
    printf("  GPU 0: %.4f ms = %.0f GB/s = %.2f%% spec\n", best0, g0, g0/7672*100);
    printf("  GPU 1: %.4f ms = %.0f GB/s = %.2f%% spec\n", best1, g1, g1/7672*100);
    printf("  Aggregate: %.0f GB/s = %.2f%% of 2× spec\n", g0+g1, (g0+g1)/15344*100);
    printf("  Each GPU achieves single-GPU ceiling? %s\n",
           (g0 > 7000 && g1 > 7000) ? "YES — independent" : "NO — interaction");

    return 0;
}
