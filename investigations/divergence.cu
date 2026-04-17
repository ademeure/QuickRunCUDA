// Thread divergence cost on B300
#include <cuda_runtime.h>
#include <cstdio>

template<int N_PATHS>
__global__ void diverge(unsigned *out, int iters, unsigned salt) {
    unsigned a = threadIdx.x + 1;
    int my_path = threadIdx.x % N_PATHS;  // each thread takes a different path

    for (int i = 0; i < iters; i++) {
        if (my_path == 0) { a = a*1.0001f + 0.0001f; }
        else if (my_path == 1) { a = a*1.0002f + 0.0002f; }
        else if (my_path == 2) { a = a*1.0003f + 0.0003f; }
        else if (my_path == 3) { a = a*1.0004f + 0.0004f; }
        else if (my_path == 4) { a = a*1.0005f + 0.0005f; }
        else if (my_path == 5) { a = a*1.0006f + 0.0006f; }
        else if (my_path == 6) { a = a*1.0007f + 0.0007f; }
        else if (my_path == 7) { a = a*1.0008f + 0.0008f; }
        else if (my_path == 8) { a = a*1.0009f + 0.0009f; }
        else if (my_path == 9) { a = a*1.0010f + 0.0010f; }
        else if (my_path == 10) { a = a*1.0011f + 0.0011f; }
        else if (my_path == 11) { a = a*1.0012f + 0.0012f; }
        else if (my_path == 12) { a = a*1.0013f + 0.0013f; }
        else if (my_path == 13) { a = a*1.0014f + 0.0014f; }
        else if (my_path == 14) { a = a*1.0015f + 0.0015f; }
        else { a = a*1.0016f + 0.0016f; }
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

__global__ void no_diverge(unsigned *out, int iters, unsigned salt) {
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        a = a*1.0001f + 0.0001f;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 1024*sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;

    auto bench = [&](auto launch, const char *name) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        printf("  %-30s %8.3f ms\n", name, best);
        return best;
    };

    printf("# B300 thread divergence cost\n");
    printf("# 148 blocks × 128 threads × 100k iters of FMA\n\n");

    float t_none = bench([&]{ no_diverge<<<blocks, threads>>>(d_out, iters, 1); }, "no divergence (1 path)");
    float t_2 = bench([&]{ diverge<2><<<blocks, threads>>>(d_out, iters, 1); }, "2 divergent paths");
    float t_4 = bench([&]{ diverge<4><<<blocks, threads>>>(d_out, iters, 1); }, "4 divergent paths");
    float t_8 = bench([&]{ diverge<8><<<blocks, threads>>>(d_out, iters, 1); }, "8 divergent paths");
    float t_16 = bench([&]{ diverge<16><<<blocks, threads>>>(d_out, iters, 1); }, "16 divergent paths");
    float t_32 = bench([&]{ diverge<32><<<blocks, threads>>>(d_out, iters, 1); }, "32 divergent paths (full)");

    printf("\n# Slowdown vs no-divergence baseline:\n");
    printf("  2 paths:  %.2fx\n", t_2 / t_none);
    printf("  4 paths:  %.2fx\n", t_4 / t_none);
    printf("  8 paths:  %.2fx\n", t_8 / t_none);
    printf("  16 paths: %.2fx\n", t_16 / t_none);
    printf("  32 paths: %.2fx\n", t_32 / t_none);

    return 0;
}
