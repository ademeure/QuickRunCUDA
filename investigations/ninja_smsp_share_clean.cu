// Properly test 1 SMSP per SM by using 148 blocks × ≤4 warps
#include <cuda_runtime.h>
#include <cstdio>

template <int N_ACTIVE_WARPS> __launch_bounds__(128, 1) __global__ void shfl_active(int *out, int N) {
    int warp_id = threadIdx.x >> 5;
    if (warp_id >= N_ACTIVE_WARPS) return;
    int lane = threadIdx.x & 31;
    int v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = lane + i;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            v[j] = __shfl_xor_sync(0xffffffff, v[j], 1);
        }
    }
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) sum += v[i];
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

template <int N_ACTIVE_WARPS> __launch_bounds__(128, 1) __global__ void atoms_active(int *out, int N) {
    __shared__ int smem[64];
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    if (warp_id >= N_ACTIVE_WARPS) return;
    for (int i = 0; i < N; i++) atomicAdd_block(&smem[0], 1);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148;  // EXACTLY 1 block per SM
    int threads = 128;
    int N = 100000;

    auto run = [&](const char* name, auto kfn, int n_active, int ilp) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_out, N);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR\n"); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_out, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * (n_active * 32) * N * ilp;
        double T = ops / (best/1000.0) / 1e9;
        // Per-SM total (since 1 block per SM)
        double per_sm_per_cy = T / 148 / 2.032;
        // Per ACTIVE SMSP (= 148 SMs * n_active SMSPs)
        double per_active_smsp_per_cy = T / (148 * n_active) / 2.032;
        printf("  %-30s warps_per_blk=%d (=%d SMSPs/SM)  %.3f ms  T=%.0f Gops  per-SM-cy=%.2f  per-SMSP-cy=%.2f\n",
            name, n_active, n_active, best, T, per_sm_per_cy, per_active_smsp_per_cy);
    };

    printf("# 1 block per SM (148 blocks total). N active warps = N SMSPs per SM\n\n");
    
    printf("SHFL (ILP=8):\n");
    run("shfl 1 warp/blk = 1 SMSP", shfl_active<1>, 1, 8);
    run("shfl 2 warps/blk = 2 SMSPs", shfl_active<2>, 2, 8);
    run("shfl 4 warps/blk = 4 SMSPs", shfl_active<4>, 4, 8);

    printf("\nATOMS.POPC.INC:\n");
    run("atoms 1 warp/blk = 1 SMSP", atoms_active<1>, 1, 1);
    run("atoms 2 warps/blk = 2 SMSPs", atoms_active<2>, 2, 1);
    run("atoms 4 warps/blk = 4 SMSPs", atoms_active<4>, 4, 1);
    return 0;
}
