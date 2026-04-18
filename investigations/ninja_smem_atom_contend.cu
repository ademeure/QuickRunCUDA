// SMEM atomic with varying contention pattern
#include <cuda_runtime.h>
#include <cstdio>

template <int N_DISTINCT>
__launch_bounds__(256, 8) __global__ void smem_atom_n(int *out, int N_iters) {
    __shared__ int smem[64];
    int lane = threadIdx.x & 31;
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    int idx = lane % N_DISTINCT;
    for (int i = 0; i < N_iters; i++) {
        atomicAdd_block(&smem[idx], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

int main() {
    cudaSetDevice(0);
    int N_iters = 10000;
    int *d_out; cudaMalloc(&d_out, 4 * 1024 * 1024); cudaMemset(d_out, 0, 4 * 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run = [&](const char* name, void(*kfn)(int*, int), int n_distinct) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_out, N_iters);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_out, N_iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        double T = ops / (best/1000.0) / 1e9;
        // SMEM is per-SM; per-SM rate
        double T_per_sm = T / 148;
        // Per SM clock 2032 MHz
        double T_per_sm_per_cy = T_per_sm / 2.032;
        double payload_gbs = ops * 4.0 / (best/1000.0) / 1e9;
        printf("  %-25s %.3f ms  T=%.0f Gops  per-SM=%.2f Gops/s  per-SM-cy=%.2f ops  payload %.0f GB/s\n",
            name, best, T, T_per_sm, T_per_sm_per_cy, payload_gbs);
    };

    printf("# SMEM atomicAdd_block contention sweep (per-block SMEM)\n");
    printf("# Within warp: lane %% N selects which of N consecutive ints to atomicAdd\n\n");
    run("smem n=1 (full popc-merge)",  smem_atom_n<1>,  1);
    run("smem n=2",  smem_atom_n<2>,  2);
    run("smem n=4",  smem_atom_n<4>,  4);
    run("smem n=8",  smem_atom_n<8>,  8);
    run("smem n=16", smem_atom_n<16>, 16);
    run("smem n=32 (no conflict, all banks)", smem_atom_n<32>, 32);

    return 0;
}
