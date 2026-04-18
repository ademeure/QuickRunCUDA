// Aggregate __shfl_sync throughput on B300
#include <cuda_runtime.h>
#include <cstdio>

template <int ILP> __launch_bounds__(256, 8) __global__ void shfl_loop(int *out, int N) {
    int v[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++) v[i] = threadIdx.x + i;
    int lane = threadIdx.x & 31;
    
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            v[j] = __shfl_sync(0xffffffff, v[j], (lane + 1) & 31);
        }
    }
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < ILP; i++) sum += v[i];
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

template <int ILP> __launch_bounds__(256, 8) __global__ void shfl_xor_loop(int *out, int N) {
    int v[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++) v[i] = threadIdx.x + i;
    
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            v[j] = __shfl_xor_sync(0xffffffff, v[j], 1);
        }
    }
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < ILP; i++) sum += v[i];
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256, N = 100000;

    auto run = [&](const char* name, auto kfn, int ilp) {
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
        long ops = (long)blocks * threads * N * ilp;
        double T = ops / (best/1000.0) / 1e9;
        double per_warp_per_cy = T / (148 * 4) / 2.032;  // per SMSP per cy
        printf("  %-25s ILP=%d  %.3f ms  T=%.0f Gops  per-SMSP-cy=%.2f\n",
            name, ilp, best, T, per_warp_per_cy);
    };

    printf("# __shfl_sync aggregate throughput\n\n");
    printf("idx-shfl (per-lane variable):\n");
    run("shfl_idx",  shfl_loop<1>, 1);
    run("shfl_idx",  shfl_loop<4>, 4);
    run("shfl_idx",  shfl_loop<8>, 8);
    run("shfl_idx",  shfl_loop<16>, 16);
    printf("\nshfl_xor (constant 1):\n");
    run("shfl_xor",  shfl_xor_loop<1>, 1);
    run("shfl_xor",  shfl_xor_loop<4>, 4);
    run("shfl_xor",  shfl_xor_loop<8>, 8);
    run("shfl_xor",  shfl_xor_loop<16>, 16);
    return 0;
}
