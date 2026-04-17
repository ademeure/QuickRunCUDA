// Measure GPU clock under different load conditions
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void measure_clock(unsigned long long *out, int delay_iters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0_clk = clock64();
        auto t0_wall = clock();  // CPU clock won't help, use clock64 only
        // Fixed delay
        for (int i = 0; i < delay_iters; i++) {
            asm volatile("mov.u32 %0, %0;" : "+r"(i));
            asm volatile("mov.u32 %0, %0;" : "+r"(i));
            asm volatile("mov.u32 %0, %0;" : "+r"(i));
            asm volatile("mov.u32 %0, %0;" : "+r"(i));
        }
        unsigned long long t1_clk = clock64();
        out[0] = t1_clk - t0_clk;
    }
}

extern "C" __global__ void heavy_load(unsigned *out, int iters) {
    float a = 1.0f + threadIdx.x * 0.001f;
    float b = 2.0f + threadIdx.x * 0.002f;
    float c = 3.0f + threadIdx.x * 0.003f;
    float d = 4.0f + threadIdx.x * 0.004f;
    for (int i = 0; i < iters; i++) {
        a = a*1.0001f + 0.0001f;
        b = b*1.0002f + 0.0002f;
        c = c*1.0003f + 0.0003f;
        d = d*1.0004f + 0.0004f;
    }
    if (a+b+c+d == 0xdeadbeef) out[blockIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out; cudaMalloc(&d_out, 16*sizeof(unsigned long long));
    unsigned *d_dummy; cudaMalloc(&d_dummy, 1024*sizeof(unsigned));

    auto measure = [&](const char *label) {
        // Run measurement kernel - measure clock cycles per nop loop
        int delay = 100000;
        measure_clock<<<1, 32>>>(d_out, delay);
        cudaDeviceSynchronize();
        unsigned long long cyc;
        cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        // Each iter has 4 nops; 100000 iters; 1 cycle per nop ideally
        // So total cycles should be ~ 100000 × 4 = 400000 cyc
        // Actual: cycles per "iter" = cyc/100000
        double cyc_per_iter = (double)cyc / delay;
        printf("  %-30s: %llu cyc / %d nop-iters = %.3f cyc/iter\n",
               label, (unsigned long long)cyc, delay, cyc_per_iter);
    };

    printf("# B300 clock state probing\n\n");

    measure("idle (cold start)");

    // Heat up GPU
    for (int i = 0; i < 5; i++) {
        heavy_load<<<148, 256>>>(d_dummy, 1000000);
        cudaDeviceSynchronize();
    }
    measure("after heavy load (1 round)");

    // Sustained load for 5 seconds
    auto t0 = std::chrono::high_resolution_clock::now();
    while (true) {
        heavy_load<<<148, 256>>>(d_dummy, 1000000);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float s = std::chrono::duration<float>(t1-t0).count();
        if (s > 5) break;
    }
    measure("after 5s sustained");

    // Use nvidia-smi to get clock
    system("nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,clocks.current.graphics --format=csv -i 0");

    return 0;
}
