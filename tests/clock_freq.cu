// Verify B300 actual clock frequency under load
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void measure_clock(unsigned long long *clk_diffs, int N) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned long long globaltimer_start, globaltimer_end;
    unsigned long long clock64_start, clock64_end;

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(globaltimer_start));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock64_start));

    // Spin for N iterations of FFMA
    float a = 1.0f;
    for (int i = 0; i < N; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(globaltimer_end));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock64_end));

    clk_diffs[0] = globaltimer_end - globaltimer_start;  // ns
    clk_diffs[1] = clock64_end - clock64_start;  // SM cycles

    // Use a to defeat DCE
    if (a > 1e30f) clk_diffs[2] = (unsigned long long)a;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    int clk_attr;
    cudaDeviceGetAttribute(&clk_attr, cudaDevAttrClockRate, 0);
    printf("# Reported clock rate: %d MHz\n\n", clk_attr / 1000);

    unsigned long long *d_diffs;
    cudaMalloc(&d_diffs, 8 * 8);

    int N_arr[] = {1000, 10000, 100000, 1000000};
    for (int N : N_arr) {
        // warmup
        for (int i = 0; i < 3; i++) {
            measure_clock<<<1, 32>>>(d_diffs, N);
            cudaDeviceSynchronize();
        }

        // Real measurement
        measure_clock<<<1, 32>>>(d_diffs, N);
        cudaDeviceSynchronize();
        unsigned long long h[3];
        cudaMemcpy(h, d_diffs, 3 * 8, cudaMemcpyDeviceToHost);

        double freq_mhz = (double)h[1] / (double)h[0] * 1000.0;
        printf("N=%-8d : clock64=%-12llu cy, globaltimer=%-12llu ns -> freq=%.2f MHz\n",
               N, h[1], h[0], freq_mhz);
    }

    cudaFree(d_diffs);
    return 0;
}
