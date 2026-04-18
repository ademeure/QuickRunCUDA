// Test ATOMS.POPC.INC.32 with various bank-conflict patterns
// SMEM has 32 banks of 4B each. addr mod 32 selects bank.
#include <cuda_runtime.h>
#include <cstdio>

// addr formula: (lane * STRIDE_INTS) % N_DISTINCT
// STRIDE=1: 8 lanes -> 8 distinct ints in 8 distinct banks (no conflict)
// STRIDE=32: 8 lanes -> 8 distinct ints all in same bank 0 (8-way conflict)
// STRIDE=16: 8 lanes -> bank 0,16,0,16,... (2-way at most)

template <int N_DISTINCT, int STRIDE_INTS>
__launch_bounds__(256, 8) __global__ void smem_add_n_stride(int *out, int N_iters) {
    __shared__ int smem[1024];  // big enough for stride=32 * n=32
    if (threadIdx.x < 1024) smem[threadIdx.x] = 0;
    __syncthreads();
    int lane = threadIdx.x & 31;
    int idx = (lane % N_DISTINCT) * STRIDE_INTS;
    for (int i = 0; i < N_iters; i++) {
        atomicAdd_block(&smem[idx], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = smem[0];
}

int main() {
    cudaSetDevice(0);
    int N_iters = 10000;
    int *d_out; cudaMalloc(&d_out, 1024 * 1024); cudaMemset(d_out, 0, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;
    int total_warps_iter = blocks * threads / 32 * N_iters;

    auto run = [&](const char* name, auto kfn, int n, int stride, int n_banks_used) {
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
        double per_warp_ns = best * 1e6 / total_warps_iter;
        double per_warp_cy = per_warp_ns * 2.032;
        printf("  %-40s n=%2d stride=%2d (%2d banks)  %.3f ms  T=%5.0f Gops  per-SM-cy=%.2f  per-warp=%.2f cy\n",
            name, n, stride, n_banks_used, best, T, T_per_sm_per_cy, per_warp_cy);
    };

    printf("# ATOMS.POPC.INC.32 with bank-conflict patterns\n");
    printf("# (atomicAdd_block of 1 - special HW POPC.INC opcode)\n\n");
    printf("n=8 (8 distinct addresses):\n");
    run("stride=1  (8 banks)",  smem_add_n_stride<8, 1>,  8, 1, 8);   // banks 0-7, no conflict
    run("stride=2  (4 banks)",  smem_add_n_stride<8, 2>,  8, 2, 4);   // banks 0,2,4,6,0,2,4,6 = 4 unique
    run("stride=4  (2 banks)",  smem_add_n_stride<8, 4>,  8, 4, 2);   // banks 0,4,8,12,16,20,24,28 = 8 banks 
    run("stride=8  (1 bank)",   smem_add_n_stride<8, 8>,  8, 8, 1);   // banks 0,8,16,24,0,8,16,24 = 4 banks
    run("stride=16 (1 bank)",   smem_add_n_stride<8, 16>, 8, 16, 1);  // banks 0,16,0,16,... = 2 banks  
    run("stride=32 (1 bank)",   smem_add_n_stride<8, 32>, 8, 32, 1);  // all same bank 0
    
    printf("\nn=4 distinct:\n");
    run("stride=1 (4 banks)",   smem_add_n_stride<4, 1>,  4, 1, 4);
    run("stride=8 (4 banks)",   smem_add_n_stride<4, 8>,  4, 8, 4);
    run("stride=32 (1 bank)",   smem_add_n_stride<4, 32>, 4, 32, 1);
    
    printf("\nn=32 distinct:\n");
    run("stride=1 (32 banks)",  smem_add_n_stride<32, 1>, 32, 1, 32);
    run("stride=2 (16 banks)",  smem_add_n_stride<32, 2>, 32, 2, 16);
    run("stride=32 (1 bank)",   smem_add_n_stride<32, 32>, 32, 32, 1);

    return 0;
}
