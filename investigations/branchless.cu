// Branchless vs branched code patterns
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void with_branch(unsigned *out, int iters, unsigned k) {
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        // Branched ternary
        if ((a + i) & 1) a = a * k + 1;
        else a = a * (k+1) + 2;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

extern "C" __global__ void branchless_select(unsigned *out, int iters, unsigned k) {
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        // Branchless: compute both, then select
        unsigned a1 = a * k + 1;
        unsigned a2 = a * (k+1) + 2;
        a = ((a + i) & 1) ? a1 : a2;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

extern "C" __global__ void branchless_arith(unsigned *out, int iters, unsigned k) {
    unsigned a = threadIdx.x + 1;
    for (int i = 0; i < iters; i++) {
        // Bit-trick: avoid both compute paths
        unsigned mask = -((a + i) & 1);  // 0 or all-1s
        unsigned d_k = (k & mask) | ((k+1) & ~mask);
        unsigned d_c = (1 & mask) | (2 & ~mask);
        a = a * d_k + d_c;
    }
    if (a == 0xdeadbeef) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 1024 * sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;
    unsigned k = 17;

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
        printf("  %-30s %.3f ms\n", name, best);
    };

    printf("# B300 branched vs branchless code patterns\n");
    printf("# 148 blocks × 128 threads × 100k iter, alternating per iter (fully divergent)\n\n");

    bench([&]{ with_branch<<<blocks, threads>>>(d_out, iters, k); }, "if-else branch");
    bench([&]{ branchless_select<<<blocks, threads>>>(d_out, iters, k); }, "ternary select");
    bench([&]{ branchless_arith<<<blocks, threads>>>(d_out, iters, k); }, "bit-mask arith");

    return 0;
}
