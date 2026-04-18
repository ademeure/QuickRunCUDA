// Test if SHFL/REDUX/ATOMS rates are per-SMSP or per-SM (shared)
// 128-thread block = 4 warps = 1 per SMSP (warp warp_id 0..3 -> SMSP 0..3)
// Activate only warp 0 (1 SMSP), warps 0+1 (2 SMSPs), or all 4 warps
#include <cuda_runtime.h>
#include <cstdio>

template <int N_ACTIVE_WARPS> __launch_bounds__(128, 16) __global__ void shfl_partial(int *out, int N) {
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

template <int N_ACTIVE_WARPS> __launch_bounds__(128, 16) __global__ void redux_partial(int *out, int N) {
    int warp_id = threadIdx.x >> 5;
    if (warp_id >= N_ACTIVE_WARPS) return;
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

template <int N_ACTIVE_WARPS> __launch_bounds__(128, 16) __global__ void atoms_partial(int *out, int N) {
    __shared__ int smem[64];
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    if (warp_id >= N_ACTIVE_WARPS) return;
    for (int i = 0; i < N; i++) atomicAdd_block(&smem[0], 1);  // POPC.INC fast path
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 16, threads = 128;  // 16 blocks/SM = 16 warps/SM = 4/SMSP, all SMSPs busy
    int N = 50000;

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
        // Per active SMSP per cy
        double per_smsp_cy = T / (148 * n_active) / 2.032;
        printf("  %-25s warps_active=%d  %.3f ms  T=%.0f Gops  per-SMSP-cy=%.2f\n",
            name, n_active, best, T, per_smsp_cy);
    };

    printf("# Per-SMSP vs per-SM rate test (128-thr blocks, 4 warps each = 1/SMSP)\n");
    printf("# Activate 1, 2, or 4 warps per block (= 1, 2, or 4 SMSPs busy per SM)\n");
    printf("# 16 blocks/SM, all blocks together saturate all 4 SMSPs\n\n");
    
    printf("SHFL (8 ILP):\n");
    run("shfl 1 warp/blk",  shfl_partial<1>, 1, 8);
    run("shfl 2 warps/blk", shfl_partial<2>, 2, 8);
    run("shfl 4 warps/blk", shfl_partial<4>, 4, 8);
    
    printf("\nREDUX.SUM (8 ILP):\n");
    run("redux 1 warp/blk", redux_partial<1>, 1, 8);
    run("redux 2 warps/blk", redux_partial<2>, 2, 8);
    run("redux 4 warps/blk", redux_partial<4>, 4, 8);

    printf("\nATOMS.POPC.INC (smem add-by-1):\n");
    run("atoms 1 warp/blk", atoms_partial<1>, 1, 1);
    run("atoms 2 warps/blk", atoms_partial<2>, 2, 1);
    run("atoms 4 warps/blk", atoms_partial<4>, 4, 1);
    return 0;
}
