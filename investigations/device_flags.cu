// cudaSetDeviceFlags effects on sync behavior
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void busy(unsigned long long *out, int cycles) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
        out[0] = clock64() - t0;
    }
}

int main(int argc, char **argv) {
    cudaSetDevice(0);

    // Set device flags BEFORE first context use
    unsigned int flags = 0;
    if (argc > 1) flags = atoi(argv[1]);

    cudaError_t err = cudaSetDeviceFlags(flags);
    printf("# cudaSetDeviceFlags(%u): %s\n", flags, cudaGetErrorString(err));

    // Print flag meaning
    const char *flag_names[] = {"Auto", "Spin", "Yield", "BlockingSync", "?", "?"};
    for (int i = 0; i < 4; i++) {
        if (flags & (1 << i)) printf("  Flag bit %d: %s\n", i, flag_names[i]);
    }
    if (flags == 0) printf("  Mode: Auto (driver chooses)\n");

    cudaStream_t s; cudaStreamCreate(&s);
    unsigned long long *d_out; cudaMalloc(&d_out, sizeof(unsigned long long));

    // Warm up
    for (int i = 0; i < 5; i++) busy<<<1, 32, 0, s>>>(d_out, 5 * 2032);  // 5us
    cudaStreamSynchronize(s);

    // Test sync overhead with various kernel runtimes
    printf("\n# Sync wall time vs kernel runtime:\n");
    printf("# %-15s %-15s %-15s\n", "kernel_us", "wall_us", "overhead_us");

    for (int us : {1, 10, 100, 1000, 10000}) {
        int cycles = us * 2032;
        // Best of 30
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            busy<<<1, 32, 0, s>>>(d_out, cycles);
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float wall_us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (wall_us < best) best = wall_us;
        }
        printf("  %-15d %-15.1f %-15.1f\n", us, best, best - us);
    }

    return 0;
}
