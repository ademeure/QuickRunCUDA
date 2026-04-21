// V10: grid.sync() cost vs __syncthreads() (standalone, cooperative launch)
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

namespace cg = cooperative_groups;

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define CHAIN_LEN 1000

__global__ void kernel_grid_sync(unsigned long long* out) {
    cg::grid_group grid = cg::this_grid();
    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    grid.sync();

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
        grid.sync();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        out[0] = t1 - t0;
    }
}

__global__ void kernel_syncthreads(unsigned long long* out) {
    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    __syncthreads();

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
        __syncthreads();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        out[0] = t1 - t0;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 16));

    // Test grid.sync with cooperative launch
    void* args[] = {&d_out};
    int blocks = 148;
    int threads = 128;

    CK(cudaLaunchCooperativeKernel((void*)kernel_grid_sync,
        dim3(blocks), dim3(threads), args, 0, 0));
    CK(cudaDeviceSynchronize());

    unsigned long long cycles_grid;
    cudaMemcpy(&cycles_grid, d_out, 8, cudaMemcpyDeviceToHost);

    // Test syncthreads (regular launch)
    kernel_syncthreads<<<blocks, threads>>>(d_out);
    CK(cudaDeviceSynchronize());

    unsigned long long cycles_sync;
    cudaMemcpy(&cycles_sync, d_out, 8, cudaMemcpyDeviceToHost);

    printf("=== Barrier cost comparison (148 blocks × 128 thr, 1001 barriers) ===\n");
    printf("  __syncthreads (per-block):  %llu cy total → %.1f cy/call\n",
           cycles_sync, (double)cycles_sync / 1001);
    printf("  grid.sync (cooperative):    %llu cy total → %.1f cy/call\n",
           cycles_grid, (double)cycles_grid / 1001);
    printf("  grid.sync / syncthreads ratio: %.2fx\n",
           (double)cycles_grid / cycles_sync);

    cudaFree(d_out);
    return 0;
}
