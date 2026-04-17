// Warp scheduler behavior on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void warps_only(unsigned long long *out, int iters) {
    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    float a = 1.0f + threadIdx.x * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (threadIdx.x == 0 && a > -42.0f) out[blockIdx.x] = end - start;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    unsigned long long *d_out;
    cudaMalloc(&d_out, prop.multiProcessorCount * sizeof(unsigned long long));

    int iters = 1000;
    int thread_arr[] = {1, 16, 32, 64, 128, 256, 512, 1024};

    printf("# B300 warp scheduling: kernel cycles vs threads/block (1 block)\n");
    printf("# All threads doing same FFMA chain, %d iters\n\n", iters);
    printf("# %-10s %-10s %-12s %-15s\n", "threads", "warps", "cycles", "cycles/warp/iter");

    for (int t : thread_arr) {
        warps_only<<<1, t>>>(d_out, iters);
        cudaDeviceSynchronize();
        unsigned long long h_cycles[2];
        cudaMemcpy(h_cycles, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        int warps = (t + 31) / 32;
        double cy_per_warp_iter = (double)h_cycles[0] / warps / iters;
        printf("  %-10d %-10d %-12llu %-15.3f\n", t, warps, h_cycles[0], cy_per_warp_iter);
    }

    cudaFree(d_out);
    return 0;
}
