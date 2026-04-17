// nanosleep precision and overhead
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void measure_nanosleep(uint64_t *out, uint32_t ns_request) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint64_t t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        asm volatile("nanosleep.u32 %0;" :: "r"(ns_request));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        out[0] = t1 - t0;
    }
}

extern "C" __global__ void measure_baseline(uint64_t *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint64_t t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        out[0] = t1 - t0;
    }
}

int main() {
    cudaSetDevice(0);
    uint64_t *d_out;
    cudaMalloc(&d_out, sizeof(uint64_t));

    int clk_mhz;
    cudaDeviceGetAttribute(&clk_mhz, cudaDevAttrClockRate, 0);
    printf("# B300 clock: %.3f MHz (from cudaDevAttrClockRate, this is BASE not boost)\n", clk_mhz/1000.0);

    // Measure observed clock by self-timing
    measure_baseline<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    uint64_t baseline; cudaMemcpy(&baseline, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("# Baseline (clock64-clock64): %llu cycles\n", (unsigned long long)baseline);

    printf("\n# nanosleep.u32 actual delay vs requested\n");
    printf("# %-15s %-15s %-15s %-15s\n",
           "request_ns", "cycles", "ns_at_2032M", "overhead_ns");

    for (uint32_t ns : {0u, 100u, 500u, 1000u, 5000u, 10000u, 100000u, 1000000u}) {
        // Best of 10
        uint64_t best = ~0ull;
        for (int t = 0; t < 10; t++) {
            measure_nanosleep<<<1, 32>>>(d_out, ns);
            cudaDeviceSynchronize();
            uint64_t cycles; cudaMemcpy(&cycles, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            if (cycles < best) best = cycles;
        }
        double ns_observed = (double)best / 2032.0 * 1000;
        double overhead = ns_observed - ns;
        printf("  %-15u %-15llu %-15.0f %-15.0f\n",
               ns, (unsigned long long)best, ns_observed, overhead);
    }

    return 0;
}
