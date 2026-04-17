// Atomic contention scaling: 1 to all threads on same vs different counters
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void atom_same(unsigned *target, int iters) {
    if (threadIdx.x < blockDim.x) {
        for (int i = 0; i < iters; i++) {
            atomicAdd(target, 1);
        }
    }
}

extern "C" __global__ void atom_per_thread(unsigned *target, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iters; i++) {
        atomicAdd(target + tid, 1);
    }
}

extern "C" __global__ void atom_per_block(unsigned *target, int iters) {
    for (int i = 0; i < iters; i++) {
        atomicAdd(target + blockIdx.x, 1);
    }
}

extern "C" __global__ void atom_per_warp(unsigned *target, int iters) {
    int wid = blockIdx.x * (blockDim.x/32) + threadIdx.x/32;
    for (int i = 0; i < iters; i++) {
        atomicAdd(target + wid, 1);
    }
}

int main() {
    cudaSetDevice(0);

    int total_threads = 148 * 128;
    unsigned *d_target;
    cudaMalloc(&d_target, total_threads * sizeof(unsigned));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 1000;
    int blocks = 148, threads = 128;
    long total_atoms = (long)blocks * threads * iters;

    auto bench = [&](auto launch, const char *name) {
        cudaMemset(d_target, 0, total_threads * sizeof(unsigned));
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            cudaMemset(d_target, 0, total_threads * sizeof(unsigned));
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double gops = total_atoms / (best/1000.0) / 1e9;
        printf("  %-30s %.3f ms  %.2f Gatomic/s\n", name, best, gops);
    };

    printf("# B300 atomic contention scaling\n");
    printf("# 148 × 128 threads × 1000 iter = %ld total atomic ops\n\n", total_atoms);

    bench([&]{ atom_same<<<blocks, threads>>>(d_target, iters); },
          "all → same address (max contend)");
    bench([&]{ atom_per_block<<<blocks, threads>>>(d_target, iters); },
          "per-block (148 targets)");
    bench([&]{ atom_per_warp<<<blocks, threads>>>(d_target, iters); },
          "per-warp (148*4=592 targets)");
    bench([&]{ atom_per_thread<<<blocks, threads>>>(d_target, iters); },
          "per-thread (no contention)");

    return 0;
}
