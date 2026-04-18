// E4 RIGOR: L2 atomic units count via stride sweep
// THEORETICAL: B300 L2 has 2 partitions × N atomic units each.
// As we spread atomic targets across more units, throughput should ramp up
// linearly until all units are busy, then plateau.
//
// Test: many warps doing atomics; stride of target addresses sweeps from
// 1 (collide all in one cache line) to 4096 B (well-separated to hit
// different L2 atomic processors).

#include <cuda_runtime.h>
#include <cstdio>

#ifndef ITERS
#define ITERS 200
#endif

extern "C" __launch_bounds__(256, 8) __global__ void stride_atomic(
    unsigned *p, int stride_bytes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned *target = (unsigned*)((char*)p + (tid * stride_bytes) % (256 * 1024 * 1024));
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        atomicAdd(target, 1);
    }
}

int main() {
    cudaSetDevice(0);
    unsigned *d_p; cudaMalloc(&d_p, 512ull * 1024 * 1024);
    cudaMemset(d_p, 0, 512ull * 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int blocks = 148, threads = 256;

    auto bench = [&](int stride_bytes) {
        for (int i = 0; i < 3; i++) stride_atomic<<<blocks, threads>>>(d_p, stride_bytes);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            stride_atomic<<<blocks, threads>>>(d_p, stride_bytes);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * ITERS;
        double gops = ops / (best/1000) / 1e9;
        return gops;
    };

    printf("# Atomic throughput vs stride (148 blocks × 256 thr × %d iters)\n", ITERS);
    printf("# stride(B)   throughput(Gops/s)\n");
    int strides[] = {0, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 262144, 1048576};
    for (int s : strides) {
        double gops = bench(s);
        printf("  %8d   %.2f\n", s, gops);
    }
    return 0;
}
