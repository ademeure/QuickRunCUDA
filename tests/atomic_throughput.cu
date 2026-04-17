// Atomic throughput on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 1000

extern "C" __global__ void atomic_local(unsigned long long *out, int reps) {
    extern __shared__ unsigned long long counter[];
    int tid = threadIdx.x;
    if (tid == 0) counter[0] = 0;
    __syncthreads();

    for (int i = 0; i < reps; i++)
        atomicAdd(counter, 1);

    __syncthreads();
    if (tid == 0) out[blockIdx.x] = counter[0];
}

extern "C" __global__ void atomic_global_same(unsigned long long *counter, int reps) {
    for (int i = 0; i < reps; i++)
        atomicAdd(counter, 1);
}

extern "C" __global__ void atomic_global_perthread(unsigned long long *counters, int reps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < reps; i++)
        atomicAdd(&counters[tid], 1);
}

extern "C" __global__ void atomic_global_perwarp(unsigned long long *counters, int reps) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    for (int i = 0; i < reps; i++)
        atomicAdd(&counters[wid], 1);
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount, threads = 128;

    unsigned long long *d_counter;
    cudaMalloc(&d_counter, blocks * threads * sizeof(unsigned long long));
    cudaMemset(d_counter, 0, blocks * threads * sizeof(unsigned long long));

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { cudaMemset(d_counter, 0, blocks*threads*8); fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            cudaMemset(d_counter, 0, blocks*threads*8);
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 atomic throughput tests\n");
    printf("# Setup: %d blocks × %d threads × %d ops each\n\n", blocks, threads, ITERS);

    long long total_ops = (long long)blocks * threads * ITERS;

    // Local atomic
    float t = bench([&]{
        atomic_local<<<blocks, threads, 8, s>>>(d_counter, ITERS);
    });
    printf("Local (shmem) atomic, all threads same:    %.3f ms = %.2f Gatomic/s\n",
           t, total_ops/(t/1e3)/1e9);

    // Global atomic, all same address
    t = bench([&]{
        atomic_global_same<<<blocks, threads, 0, s>>>(d_counter, ITERS);
    });
    printf("Global atomic, ALL %d threads same addr:  %.3f ms = %.2f Gatomic/s\n",
           blocks * threads, t, total_ops/(t/1e3)/1e9);

    // Global atomic, per-thread (no contention)
    t = bench([&]{
        atomic_global_perthread<<<blocks, threads, 0, s>>>(d_counter, ITERS);
    });
    printf("Global atomic, per-thread (no contend):    %.3f ms = %.2f Gatomic/s\n",
           t, total_ops/(t/1e3)/1e9);

    // Global atomic, per-warp (32 threads contend)
    t = bench([&]{
        atomic_global_perwarp<<<blocks, threads, 0, s>>>(d_counter, ITERS);
    });
    printf("Global atomic, per-warp (32-way contend):  %.3f ms = %.2f Gatomic/s\n",
           t, total_ops/(t/1e3)/1e9);

    cudaStreamDestroy(s);
    cudaFree(d_counter);
    return 0;
}
