// HBM atomic at varying occupancy: catch latency-hiding effects
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

template <int OCC>
__launch_bounds__(256, OCC) __global__ void atom_unique_lines(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 32) % N_addrs;
        atomicAdd(&p[addr], 1);
    }
}

template <int OCC>
double bench(int *d_p, long N_addrs, int N_iters) {
    int blocks = 148 * OCC;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) atom_unique_lines<OCC><<<blocks, 256>>>(d_p, N_iters, N_addrs);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) {
        printf("ERR OCC=%d\n", OCC);
        return 0;
    }
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        atom_unique_lines<OCC><<<blocks, 256>>>(d_p, N_iters, N_addrs);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = (long)blocks * 256 * N_iters;
    double gops = ops / (best/1000.0) / 1e9;
    double payload_gbs = ops * 4.0 / (best/1000.0) / 1e9;
    printf("  OCC=%d (%dthr/SM, %d blocks)  %.3f ms  %.1f Gops/s  payload %.0f GB/s\n",
        OCC, OCC * 256, blocks, best, gops, payload_gbs);
    return gops;
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    long WS_MB = (argc > 1) ? atol(argv[1]) : 1024;
    long N_addrs = WS_MB * 1024 * 1024 / 4;
    int *d_p; cudaMalloc(&d_p, (size_t)N_addrs * 4);
    cudaMemset(d_p, 0, (size_t)N_addrs * 4);
    int N_iters = 100;
    printf("# WS=%ldMB N_iters=%d (HBM-resident if WS>>60MB)\n", WS_MB, N_iters);
    bench<1>(d_p, N_addrs, N_iters);
    bench<2>(d_p, N_addrs, N_iters);
    bench<4>(d_p, N_addrs, N_iters);
    bench<8>(d_p, N_addrs, N_iters);  // 2048 thr/SM = full occ
    return 0;
}
