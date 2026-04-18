#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
__launch_bounds__(256, 4) __global__ void hbm_atom(int *p, int N_iters, int N_addrs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread visits N_iters DIFFERENT addresses spread across full WS
    int hash = tid * 2654435761u;
    for (int i = 0; i < N_iters; i++) {
        int addr = (hash + i * 4096) & (N_addrs - 1);  // 4 KB stride per iter
        atomicAdd(&p[addr], 1);
    }
}
int main() {
    cudaSetDevice(0);
    int N_addrs = 256 * 1024 * 1024 / 4;  // 256 MB int array
    int *d_p; cudaMalloc(&d_p, N_addrs * 4);
    cudaMemset(d_p, 0, N_addrs * 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto run = [&](int blocks, int N_iters) {
        for (int i = 0; i < 3; i++) hbm_atom<<<blocks, 256>>>(d_p, N_iters, N_addrs);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) return;
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            hbm_atom<<<blocks, 256>>>(d_p, N_iters, N_addrs);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * 256 * N_iters;
        double gops = ops / (best/1000.0) / 1e9;
        printf("  blocks=%d  N_iters=%d  %.4f ms = %.1f Gops/s atomic\n",
               blocks, N_iters, best, gops);
    };
    printf("# HBM atomic (256 MB WS, 4KB stride per iter, addresses spread)\n");
    run(148, 10000);
    run(296, 5000);
    run(592, 5000);
    run(1184, 5000);
    return 0;
}
