// C6 v3: measure cost of red.release vs red.relaxed vs plain
#include <cuda_runtime.h>
#include <cstdio>

#ifndef ITERS
#define ITERS 1000
#endif

extern "C" __global__ void red_plain(unsigned *p, int iters) {
    unsigned *q = p + (blockIdx.x * blockDim.x + threadIdx.x);
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("red.global.add.u32 [%0], 1;" :: "l"(q) : "memory");
}

extern "C" __global__ void red_relaxed(unsigned *p, int iters) {
    unsigned *q = p + (blockIdx.x * blockDim.x + threadIdx.x);
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" :: "l"(q) : "memory");
}

extern "C" __global__ void red_release(unsigned *p, int iters) {
    unsigned *q = p + (blockIdx.x * blockDim.x + threadIdx.x);
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(q) : "memory");
}

int main() {
    cudaSetDevice(0);
    int blocks = 148, threads = 256;
    unsigned *d_p; cudaMalloc(&d_p, blocks * threads * sizeof(unsigned));
    cudaMemset(d_p, 0, blocks * threads * sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch, const char* label) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 10; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total_ops = (long)blocks * threads * ITERS;
        double per_op_ns = best * 1e6 / ITERS;  // per-thread serial
        double agg_gops = total_ops / (best/1000) / 1e9;
        printf("  %s: %.4f ms, %.1f ns/op (per-thr serial), %.2f Gops/s aggregate\n",
               label, best, per_op_ns, agg_gops);
    };

    bench([&]{ red_plain<<<blocks, threads>>>(d_p, ITERS); }, "red.global             ");
    bench([&]{ red_relaxed<<<blocks, threads>>>(d_p, ITERS); }, "red.relaxed.gpu.global ");
    bench([&]{ red_release<<<blocks, threads>>>(d_p, ITERS); }, "red.release.gpu.global ");
    return 0;
}
