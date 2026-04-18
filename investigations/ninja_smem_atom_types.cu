// SMEM atomic types: ADD vs XOR vs EXCH at varying N distinct
#include <cuda_runtime.h>
#include <cstdio>

template <int N> __launch_bounds__(256, 8) __global__ void smem_add(int *out, int N_iters) {
    __shared__ int smem[64];
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    int idx = (threadIdx.x & 31) % N;
    for (int i = 0; i < N_iters; i++) atomicAdd_block(&smem[idx], 1);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

template <int N> __launch_bounds__(256, 8) __global__ void smem_xor(int *out, int N_iters) {
    __shared__ int smem[64];
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    int idx = (threadIdx.x & 31) % N;
    for (int i = 0; i < N_iters; i++) atomicXor_block(&smem[idx], 1);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

template <int N> __launch_bounds__(256, 8) __global__ void smem_exch(int *out, int N_iters) {
    __shared__ int smem[64];
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    int idx = (threadIdx.x & 31) % N;
    int v = threadIdx.x;
    for (int i = 0; i < N_iters; i++) v = atomicExch_block(&smem[idx], v + i);
    __syncthreads();
    if (v == 0xdeadbeef) out[blockIdx.x] = v;
}

int main() {
    cudaSetDevice(0);
    int N_iters = 10000;
    int *d_out; cudaMalloc(&d_out, 1024 * 1024); cudaMemset(d_out, 0, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run = [&](const char* name, auto kfn, int n) {
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
        double T_per_sm_per_cy = T / 148 / 2.032;
        printf("  %-15s n=%2d  %.3f ms  T=%5.0f Gops  per-SM-cy=%.2f ops\n",
            name, n, best, T, T_per_sm_per_cy);
    };

    printf("# SMEM atomic types vs N (using _block variants)\n\n");
    printf("ADD:\n");
    run("smem_add", smem_add<1>, 1);
    run("smem_add", smem_add<8>, 8);
    run("smem_add", smem_add<32>, 32);
    printf("\nXOR:\n");
    run("smem_xor", smem_xor<1>, 1);
    run("smem_xor", smem_xor<8>, 8);
    run("smem_xor", smem_xor<32>, 32);
    printf("\nEXCH:\n");
    run("smem_exch", smem_exch<1>, 1);
    run("smem_exch", smem_exch<8>, 8);
    run("smem_exch", smem_exch<32>, 32);
    return 0;
}
