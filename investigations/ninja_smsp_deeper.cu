// Deeper SMSP analysis: vary warps/SMSP for SHFL, test SMEM, test contention
#include <cuda_runtime.h>
#include <cstdio>

// SHFL: 1 SMSP, vary warps active on it (1, 2, 4, 8, 16 warps)
template <int N_WARPS_ON_SMSP0> __launch_bounds__(1024, 1) __global__ void shfl_warps(int *out, int N) {
    int warp_id = threadIdx.x >> 5;
    // Only SMSP 0: warps where (warp_id % 4) == 0. There are 8 such warps in 32-warp block.
    // Active = first N_WARPS_ON_SMSP0 of those (warps 0, 4, 8, ..., (N-1)*4)
    int smsp = warp_id & 3;
    int slot_on_smsp = warp_id >> 2;
    if (smsp != 0 || slot_on_smsp >= N_WARPS_ON_SMSP0) return;
    int lane = threadIdx.x & 31;
    int v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = lane + i;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) v[j] = __shfl_xor_sync(0xffffffff, v[j], 1);
    }
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) sum += v[i];
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

// SMEM LDS: per-SMSP test (similar structure to SHFL)
template <int N_ACTIVE_SMSP> __launch_bounds__(1024, 1) __global__ void smem_lds(int *out, int N) {
    __shared__ int smem[1024];
    if (threadIdx.x < 1024) smem[threadIdx.x] = threadIdx.x;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    if ((warp_id % 4) >= N_ACTIVE_SMSP) return;
    int lane = threadIdx.x & 31;
    int v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = lane;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) v[j] = smem[(v[j] + j) & 1023];  // dependent load
    }
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) sum += v[i];
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

// SHFL + SMEM mixed: do they contend?
template <int N_ACTIVE_SMSP> __launch_bounds__(1024, 1) __global__ void shfl_plus_lds(int *out, int N) {
    __shared__ int smem[1024];
    if (threadIdx.x < 1024) smem[threadIdx.x] = threadIdx.x;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    if ((warp_id % 4) >= N_ACTIVE_SMSP) return;
    int lane = threadIdx.x & 31;
    int v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = lane;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            v[j] = __shfl_xor_sync(0xffffffff, v[j], 1);  // shfl
            v[j] = smem[(v[j] + j) & 1023];               // lds
        }
    }
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) sum += v[i];
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148, threads = 1024;
    int N = 100000;

    auto run = [&](const char* name, auto kfn, int n_warps_active_per_smsp_or_smsp_count, int total_active_warps_per_block, int ilp, int n_ops_per_inner) {
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
        long ops = (long)blocks * (total_active_warps_per_block * 32) * N * ilp * n_ops_per_inner;
        double T = ops / (best/1000.0) / 1e9;
        double per_sm_cy = T / 148 / 2.032;
        printf("  %-30s param=%d  %.3f ms  T=%5.0f Gops  per-SM-cy=%.2f\n",
            name, n_warps_active_per_smsp_or_smsp_count, best, T, per_sm_cy);
    };

    printf("# TEST 1: SHFL on SMSP 0 ONLY, vary warps on it (1, 2, 4, 8, 16)\n");
    run("shfl_warps_on_smsp0", shfl_warps<1>,  1,  1,  8, 1);
    run("shfl_warps_on_smsp0", shfl_warps<2>,  2,  2,  8, 1);
    run("shfl_warps_on_smsp0", shfl_warps<4>,  4,  4,  8, 1);
    run("shfl_warps_on_smsp0", shfl_warps<8>,  8,  8,  8, 1);

    printf("\n# TEST 2: SMEM LDS dependent-chain, vary active SMSPs (8 warps each)\n");
    run("smem_lds active_SMSPs", smem_lds<1>, 1, 8,  8, 1);
    run("smem_lds active_SMSPs", smem_lds<2>, 2, 16, 8, 1);
    run("smem_lds active_SMSPs", smem_lds<4>, 4, 32, 8, 1);

    printf("\n# TEST 3: SHFL+SMEM mixed (per inner = 1 shfl + 1 lds)\n");
    run("shfl+lds active_SMSPs", shfl_plus_lds<1>, 1, 8,  8, 2);
    run("shfl+lds active_SMSPs", shfl_plus_lds<2>, 2, 16, 8, 2);
    run("shfl+lds active_SMSPs", shfl_plus_lds<4>, 4, 32, 8, 2);

    return 0;
}
