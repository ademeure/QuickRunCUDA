// Cooperative grid sync cost vs grid size
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;

__global__ void grid_sync_loop(long long *out, int N) {
    auto grid = cg::this_grid();
    long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        grid.sync();
    }
    long long t1 = clock64();
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = t1 - t0;
}

int main() {
    cudaSetDevice(0);
    long long *d_out; cudaMalloc(&d_out, 16);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    
    int N = 1000;
    printf("# Cooperative grid.sync() cost vs grid size (N=%d syncs per kernel)\n", N);
    printf("# Per-sync clock64 (block 0) and per-sync wall-clock (event)\n");
    
    for (int blocks : {32, 74, 148, 296, 592, 1184}) {
        int threads = 32;
        // Cooperative launch
        cudaLaunchAttribute attr = {};
        attr.id = cudaLaunchAttributeCooperative;
        attr.val.cooperative = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(blocks);
        cfg.blockDim = dim3(threads);
        cfg.numAttrs = 1;
        cfg.attrs = &attr;
        // Warmup
        for (int i = 0; i < 3; i++) cudaLaunchKernelEx(&cfg, grid_sync_loop, d_out, N);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
            printf("  blocks=%d: ERR %s\n", blocks, cudaGetErrorString(cudaGetLastError()));
            continue;
        }
        // Wall-clock measurement
        float best_ms = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            cudaLaunchKernelEx(&cfg, grid_sync_loop, d_out, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best_ms) best_ms = ms;
        }
        long long h; cudaMemcpy(&h, d_out, 8, cudaMemcpyDeviceToHost);
        double per_sync_ns_clock64 = (double)h / N / 2.032;  // Use SM clock 2032
        double per_sync_us_wall = best_ms * 1000.0 / N;
        printf("  blocks=%4d: clock64 %.1f cy = %.2f ns,   wall %.4f ms = %.2f us/sync\n",
               blocks, (double)h/N, per_sync_ns_clock64, best_ms, per_sync_us_wall);
    }
    return 0;
}
