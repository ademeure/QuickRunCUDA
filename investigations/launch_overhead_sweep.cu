// Kernel launch overhead - actual measured cost vs kernel runtime
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void busy_known_time(unsigned long long *out, int delay_cycles) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < delay_cycles) {}
        out[0] = clock64() - t0;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out; cudaMalloc(&d_out, sizeof(unsigned long long));
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto launch, int trials = 100) {
        for (int i = 0; i < 5; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            launch();
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 measured kernel launch overhead vs known kernel runtime\n");
    printf("# %-15s %-15s %-15s %-15s\n",
           "delay_cyc", "kernel_us", "wall_us", "overhead_us");

    for (int delay : {0, 1000, 5000, 20000, 100000, 1000000, 10000000}) {
        // Set delay
        float wall = bench([&]{
            busy_known_time<<<1, 32, 0, s>>>(d_out, delay);
        }, 30);

        // Get actual cycles spun
        unsigned long long cyc;
        cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        float kernel_us = cyc / 2.032 / 1000;
        float overhead = wall - kernel_us;

        printf("  %-15d %-15.2f %-15.2f %-15.2f\n",
               delay, kernel_us, wall, overhead);
    }

    return 0;
}
