// REDUX.SUM warp-wide reduction throughput
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
namespace cg = cooperative_groups;

template <int ILP> __launch_bounds__(256, 8) __global__ void redux_sum(int *out, int N) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int v[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++) v[i] = threadIdx.x + i;
    
    int sum_acc = 0;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            int r = cg::reduce(warp, v[j], cg::plus<int>());
            sum_acc += r;
            v[j] += i;  // mutate to defeat hoisting
        }
    }
    if (sum_acc == 0xdeadbeef) out[blockIdx.x] = sum_acc;
}

// Direct PTX redux.sum
template <int ILP> __launch_bounds__(256, 8) __global__ void redux_ptx(int *out, int N) {
    int v[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++) v[i] = threadIdx.x + i;
    int sum_acc = 0;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
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
        long ops = (long)blocks * threads * N * ilp;  // thread-level reduce calls
        double T = ops / (best/1000.0) / 1e9;
        double per_smsp_cy = T / (148 * 4) / 2.032;
        printf("  %-25s ILP=%d  %.3f ms  T=%.0f Gops  per-SMSP-cy=%.2f\n",
            name, ilp, best, T, per_smsp_cy);
    };

    printf("# REDUX.SUM aggregate throughput\n\n");
    printf("cg::reduce (warp.sync.add):\n");
    run("redux_sum", redux_sum<1>, 1);
    run("redux_sum", redux_sum<4>, 4);
    run("redux_sum", redux_sum<8>, 8);
    printf("\nDirect PTX redux.sync.add.u32:\n");
    run("redux_ptx", redux_ptx<1>, 1);
    run("redux_ptx", redux_ptx<4>, 4);
    run("redux_ptx", redux_ptx<8>, 8);
    return 0;
}
