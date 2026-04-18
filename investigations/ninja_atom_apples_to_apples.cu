// Apples-to-apples atomic compare: same access pattern, L2-resident vs HBM-resident
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Each thread iter visits a unique cache line
__launch_bounds__(256, 4) __global__ void atom_unique_lines(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 32) % N_addrs;
        atomicAdd(&p[addr], 1);
    }
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    long WS_MB = (argc > 1) ? atol(argv[1]) : 256;
    long N_addrs = WS_MB * 1024 * 1024 / 4;
    int N_iters = (argc > 2) ? atoi(argv[2]) : 100;
    int *d_p; cudaMalloc(&d_p, (size_t)N_addrs * 4);
    cudaMemset(d_p, 0, (size_t)N_addrs * 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 1184, threads = 256;

    for (int i = 0; i < 3; i++) atom_unique_lines<<<blocks, threads>>>(d_p, N_iters, N_addrs);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR\n"); return 1; }

    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        atom_unique_lines<<<blocks, threads>>>(d_p, N_iters, N_addrs);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = (long)blocks * threads * N_iters;
    double gops = ops / (best/1000.0) / 1e9;
    double payload_gbs = ops * 4.0 / (best/1000.0) / 1e9;
    double cl_traffic_gbs = ops * 128.0 / (best/1000.0) / 1e9;
    printf("WS=%4ldMB N_iters=%d ops=%.2eM:  %.3f ms  %.1f Gops/s  payload %.0f GB/s  if-CL-each %.0f GB/s\n",
        WS_MB, N_iters, ops/1e6, best, gops, payload_gbs, cl_traffic_gbs);
    return 0;
}
