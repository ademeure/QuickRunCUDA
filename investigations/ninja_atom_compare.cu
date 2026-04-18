// Proper atomic comparison: ALWAYS report Gops/s + bytes/s
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Stride-4B test (warp threads hit consecutive ints — cache-line-combining benefit)
__launch_bounds__(256, 4) __global__ void atom_stride4(int *p, int N_iters, int N_addrs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base = (tid / 32) * 32;  // each warp's 32 threads target consecutive 32 ints (one cache line)
    int lane = tid & 31;
    for (int i = 0; i < N_iters; i++) {
        int addr = (base * 4 + i * 8192 + lane) & (N_addrs - 1);  // warp-coalesced base + lane offset, varies per iter
        atomicAdd(&p[addr], 1);
    }
}

// Stride-4KB test (each thread hits different cache line — no combining)
__launch_bounds__(256, 4) __global__ void atom_stride4k(int *p, int N_iters, int N_addrs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int hash = tid * 2654435761u;
    for (int i = 0; i < N_iters; i++) {
        int addr = (hash + i * 4096) & (N_addrs - 1);
        atomicAdd(&p[addr], 1);
    }
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    int WS_MB = (argc > 1) ? atoi(argv[1]) : 256;
    int N_addrs = (size_t)WS_MB * 1024 * 1024 / 4;
    int *d_p; cudaMalloc(&d_p, (size_t)N_addrs * 4);
    cudaMemset(d_p, 0, (size_t)N_addrs * 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto run = [&](const char* name, void(*kfn)(int*, int, int), int blocks, int N_iters) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, 256>>>(d_p, N_iters, N_addrs);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) return;
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, 256>>>(d_p, N_iters, N_addrs);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * 256 * N_iters;
        double gops = ops / (best/1000.0) / 1e9;
        double payload_gbs = ops * 4.0 / (best/1000.0) / 1e9;  // 4 bytes per int atomic
        printf("  %-20s  WS=%dMB blocks=%d N=%d  %.3f ms = %.1f Gops/s = %.0f GB/s payload\n",
               name, WS_MB, blocks, N_iters, best, gops, payload_gbs);
    };

    printf("# Atomic comparison (always reports Gops/s + payload GB/s)\n");
    printf("\n# L2-territory (WS=%dMB)\n", WS_MB);
    run("stride=4B (combine)", atom_stride4, 1184, 5000);
    run("stride=4KB (no comb)", atom_stride4k, 1184, 5000);

    return 0;
}
