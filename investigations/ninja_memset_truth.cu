// Investigate the wall-clock vs ncu gap for our ninja memset
// Run multiple iters; if wall-clock outpaces DRAM rate, it's L2 absorption
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void w_ninja(int *data, int v, int per_block) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    // Each warp processes per_block KB of data
    for (int it = 0; it < per_block; it++) {
        int *p = data + ((warp_id * per_block + it) * 32 + lane) * 8;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("# wall-clock vs DRAM rate at varying buffer sizes\n");
    printf("# (with the same NINJA recipe, just varying total bytes)\n");
    printf("# bytes(MB)  per_block_KB  wall-clock GB/s\n");

    for (size_t MB : {16, 64, 256, 512, 1024, 4096, 8192, 16384}) {
        size_t bytes = MB * 1024 * 1024;
        int *d; cudaMalloc(&d, bytes);

        // Use the optimal 1-KB-per-warp recipe
        int per_block = 1;  // 1 v8 store per warp
        int n_warps = bytes / (32 * 8 * 4);  // each warp covers 1 KB
        int blocks = (n_warps + 7) / 8;

        for (int i = 0; i < 5; i++) w_ninja<<<blocks, 256>>>(d, 0xab, per_block);
        cudaDeviceSynchronize();

        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaEventRecord(e0);
            w_ninja<<<blocks, 256>>>(d, 0xab, per_block);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double gbs = bytes / (best/1000) / 1e9;
        printf("  %5zu      %5d      %6.0f GB/s = %.2f%% spec\n",
               MB, per_block, gbs, gbs/7672*100);

        cudaFree(d);
    }
    return 0;
}
