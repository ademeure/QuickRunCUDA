// Re-test OLD pattern (atom_proper VERSION A) with proper rigor
// Pattern: all warps contend on SAME N-line region with L2-cached reuse
#include <cuda_runtime.h>
#include <cstdio>

// VERSION A re-implementation: small stride per iter, WS varies
__launch_bounds__(256, 4) __global__ void atomA(int *p, int N_iters, int N_addrs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_base = (tid / 32) * 32;
    int lane = tid & 31;
    for (int i = 0; i < N_iters; i++) {
        int addr = (warp_base * 4 + i * 8192 + lane) & (N_addrs - 1);
        atomicAdd(&p[addr], 1);
    }
}

int main(int argc, char** argv) {
    cudaSetDevice(0);
    int WS_MB = (argc > 1) ? atoi(argv[1]) : 16;
    int N = (long)WS_MB * 1024 * 1024 / 4;
    int *d_p; cudaMalloc(&d_p, (size_t)N * 4); cudaMemset(d_p, 0, (size_t)N * 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 1184, threads = 256, N_iters = 5000;

    for (int i = 0; i < 3; i++) atomA<<<blocks, threads>>>(d_p, N_iters, N);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR\n"); return 1; }

    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        atomA<<<blocks, threads>>>(d_p, N_iters, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = (long)blocks * threads * N_iters;
    double T = ops / (best/1000.0) / 1e9;
    double payload_gbs = ops * 4.0 / (best/1000.0) / 1e9;
    printf("WS=%d MB  %.3f ms  T=%.0f Gops  payload %.0f GB/s = %.2f TB/s\n",
        WS_MB, best, T, payload_gbs, payload_gbs/1000);
    return 0;
}
