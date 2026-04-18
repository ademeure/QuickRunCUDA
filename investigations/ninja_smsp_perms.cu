// SHFL with specific SMSP pair combinations
// Filter mask: bit i = 1 means SMSP i active
#include <cuda_runtime.h>
#include <cstdio>

template <int SMSP_MASK> __launch_bounds__(1024, 1) __global__ void shfl_perm(int *out, int N) {
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
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

template <int SMSP_MASK> __launch_bounds__(1024, 1) __global__ void atoms_perm(int *out, int N) {
    __shared__ int smem[64];
    if (threadIdx.x < 64) smem[threadIdx.x] = 0;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    for (int i = 0; i < N; i++) atomicAdd_block(&smem[0], 1);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

int popc(int x) { return __builtin_popcount(x); }

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148, threads = 1024;
    int N = 100000;

    auto run = [&](const char* name, auto kfn, int mask, int ilp) {
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
        int n_active = popc(mask);
        long active_warps_per_block = n_active * 8;  // 8 warps/SMSP in 32-warp block
        long ops = (long)blocks * (active_warps_per_block * 32) * N * ilp;
        double T = ops / (best/1000.0) / 1e9;
        double per_sm_cy = T / 148 / 2.032;
        printf("  %-25s mask=0b%d%d%d%d (SMSPs %s%s%s%s) %.3f ms  T=%5.0f Gops  per-SM-cy=%.2f\n",
            name,
            (mask>>3)&1, (mask>>2)&1, (mask>>1)&1, mask&1,
            (mask&1)?"0":"-", (mask>>1&1)?"1":"-", (mask>>2&1)?"2":"-", (mask>>3&1)?"3":"-",
            best, T, per_sm_cy);
    };

    printf("# SHFL permutations (which SMSPs active)\n\n");
    run("shfl SMSP", shfl_perm<0b0001>, 0b0001, 8);  // 0
    run("shfl SMSP", shfl_perm<0b0010>, 0b0010, 8);  // 1
    run("shfl SMSP", shfl_perm<0b0100>, 0b0100, 8);  // 2
    run("shfl SMSP", shfl_perm<0b1000>, 0b1000, 8);  // 3
    printf("\n# 2-SMSP pairs:\n");
    run("shfl SMSP", shfl_perm<0b0011>, 0b0011, 8);  // 0+1
    run("shfl SMSP", shfl_perm<0b0101>, 0b0101, 8);  // 0+2
    run("shfl SMSP", shfl_perm<0b1001>, 0b1001, 8);  // 0+3
    run("shfl SMSP", shfl_perm<0b0110>, 0b0110, 8);  // 1+2
    run("shfl SMSP", shfl_perm<0b1010>, 0b1010, 8);  // 1+3
    run("shfl SMSP", shfl_perm<0b1100>, 0b1100, 8);  // 2+3
    printf("\n# 3-SMSP triples:\n");
    run("shfl SMSP", shfl_perm<0b0111>, 0b0111, 8);  // 0+1+2
    run("shfl SMSP", shfl_perm<0b1011>, 0b1011, 8);  // 0+1+3
    run("shfl SMSP", shfl_perm<0b1101>, 0b1101, 8);  // 0+2+3
    run("shfl SMSP", shfl_perm<0b1110>, 0b1110, 8);  // 1+2+3
    printf("\n# 4 SMSPs:\n");
    run("shfl SMSP", shfl_perm<0b1111>, 0b1111, 8);
    
    printf("\n## Same for ATOMS:\n");
    run("atoms SMSP", atoms_perm<0b0001>, 0b0001, 1);
    run("atoms SMSP", atoms_perm<0b0011>, 0b0011, 1);
    run("atoms SMSP", atoms_perm<0b0101>, 0b0101, 1);
    run("atoms SMSP", atoms_perm<0b1001>, 0b1001, 1);
    run("atoms SMSP", atoms_perm<0b1010>, 0b1010, 1);
    run("atoms SMSP", atoms_perm<0b1100>, 0b1100, 1);
    run("atoms SMSP", atoms_perm<0b1111>, 0b1111, 1);
    return 0;
}
