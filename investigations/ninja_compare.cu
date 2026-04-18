// Compare: cudaMemset vs new ninja recipe vs old v8 recipe
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void w_old(int *data, int v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *base = data + warp_id * (32 * 1024 / 4);
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = base + (it * 32 + lane) * 8;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}

// NINJA: 1 store per warp = max parallelism
__launch_bounds__(256, 8) __global__ void w_ninja(int *data, int v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *p = data + (warp_id * 32 + lane) * 8;
    asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
        :: "l"(p), "r"(v) : "memory");
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;
    int *d; cudaMalloc(&d, bytes);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch, const char* label) {
        for (int i = 0; i < 5; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double gbs = bytes / (best/1000) / 1e9;
        printf("  %s: %.4f ms = %.1f GB/s = %.2f%% spec\n",
               label, best, gbs, gbs/7672*100);
    };

    int blocks_old = bytes / (256 * 1024);    // 16384 blocks
    int blocks_ninja = bytes / (256 * 32);    // 524288 blocks (32 B per thread)
    printf("# Comparison at 4 GB workload\n");
    bench([&]{ cudaMemset(d, 0xab, bytes); }, "cudaMemset                ");
    bench([&]{ w_old<<<blocks_old, 256>>>(d, 0xab); }, "w_old (32 iter v8)        ");
    bench([&]{ w_ninja<<<blocks_ninja, 256>>>(d, 0xab); }, "w_ninja (1 store per warp)");

    return 0;
}
