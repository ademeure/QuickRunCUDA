// 1024-thread blocks (32 warps) -> 8 warps per SMSP for full latency hiding
// Filter: only activate warps where warp_id < N_ACTIVE_SMSP
//   N_ACTIVE_SMSP = 1 -> 8 warps on SMSP 0, others idle
//   N_ACTIVE_SMSP = 2 -> 8 warps on SMSPs 0+1
//   etc
// IMPORTANT: warp_id 0-3 are SMSPs 0-3; warp 4 wraps to SMSP 0; etc
// So warps with (warp_id % 4) < N_ACTIVE_SMSP are the active ones
#include <cuda_runtime.h>
#include <cstdio>

template <int N_ACTIVE_SMSP> __launch_bounds__(1024, 1) __global__ void shfl_active(int *out, int N) {
    int warp_id = threadIdx.x >> 5;
    if ((warp_id % 4) >= N_ACTIVE_SMSP) return;  // 8 warps per active SMSP
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

template <int N_ACTIVE_SMSP> __launch_bounds__(1024, 1) __global__ void atoms_active(int *out, int N) {
    __shared__ int smem[64];
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    if ((warp_id % 4) >= N_ACTIVE_SMSP) return;
    for (int i = 0; i < N; i++) atomicAdd_block(&smem[0], 1);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

template <int N_ACTIVE_SMSP> __launch_bounds__(1024, 1) __global__ void redux_active(int *out, int N) {
    int warp_id = threadIdx.x >> 5;
    if ((warp_id % 4) >= N_ACTIVE_SMSP) return;
    int v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i] = threadIdx.x + i;
    int sum_acc = 0;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int r;
            asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v[j]));
            sum_acc += r;
            v[j] += i;
        }
    }
    if (sum_acc == 0xdeadbeef) out[blockIdx.x] = sum_acc;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148, threads = 1024;  // 1 block/SM, 32 warps/block, 8 warps/SMSP for latency hiding
    int N = 100000;

    auto run = [&](const char* name, auto kfn, int n_active_smsp, int ilp) {
        int active_warps_per_block = n_active_smsp * 8;  // 8 warps per active SMSP
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
        long ops = (long)blocks * (active_warps_per_block * 32) * N * ilp;
        double T = ops / (best/1000.0) / 1e9;
        double per_sm_cy = T / 148 / 2.032;
        double per_active_smsp_cy = T / (148 * n_active_smsp) / 2.032;
        printf("  %-30s active_SMSPs=%d (8 warps each)  %.3f ms  T=%5.0f Gops  per-SM-cy=%.2f  per-SMSP-cy=%.2f\n",
            name, n_active_smsp, best, T, per_sm_cy, per_active_smsp_cy);
    };

    printf("# 1 block/SM, 1024 thr/block (32 warps), 8 warps per ACTIVE SMSP (full latency hide)\n");
    printf("# Filter: warp_id %% 4 < N_ACTIVE_SMSP\n\n");

    printf("SHFL (ILP=8):\n");
    run("shfl SMSPs=1", shfl_active<1>, 1, 8);
    run("shfl SMSPs=2", shfl_active<2>, 2, 8);
    run("shfl SMSPs=4", shfl_active<4>, 4, 8);

    printf("\nATOMS.POPC.INC:\n");
    run("atoms SMSPs=1", atoms_active<1>, 1, 1);
    run("atoms SMSPs=2", atoms_active<2>, 2, 1);
    run("atoms SMSPs=4", atoms_active<4>, 4, 1);

    printf("\nREDUX.SUM (ILP=8):\n");
    run("redux SMSPs=1", redux_active<1>, 1, 8);
    run("redux SMSPs=2", redux_active<2>, 2, 8);
    run("redux SMSPs=4", redux_active<4>, 4, 8);

    return 0;
}
