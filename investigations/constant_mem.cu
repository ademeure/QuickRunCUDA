// Constant memory broadcast vs L1 vs SHMEM throughput
#include <cuda_runtime.h>
#include <cstdio>

__constant__ float cmem[16384];  // 64 KB constant mem

__global__ void cmem_uniform_read(float *out, int iters) {
    // All threads read same address — broadcast path
    float a = 0;
    for (int i = 0; i < iters; i++) {
        a += cmem[i & 1023];
    }
    if (a < -1e30f) out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

__global__ void cmem_diverge_read(float *out, int iters, int seed) {
    // Each thread reads different address — no broadcast
    float a = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iters; i++) {
        a += cmem[(tid + i + seed) & 1023];
    }
    if (a < -1e30f) out[tid] = a;
}

__global__ void smem_uniform_read(float *out, int iters) {
    __shared__ float smem[1024];
    if (threadIdx.x < 1024) smem[threadIdx.x] = (float)threadIdx.x;
    __syncthreads();

    float a = 0;
    for (int i = 0; i < iters; i++) {
        a += smem[i & 1023];
    }
    if (a < -1e30f) out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

int main() {
    cudaSetDevice(0);

    // Init constant mem
    float h_cmem[16384];
    for (int i = 0; i < 16384; i++) h_cmem[i] = (float)i;
    cudaMemcpyToSymbol(cmem, h_cmem, sizeof(h_cmem));

    float *d_out; cudaMalloc(&d_out, 148 * 256 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 1000000;
    int blocks = 148, threads = 128;
    long total_reads = (long)blocks * threads * iters;

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
        double gops = total_reads / (best/1000.0) / 1e9;
        double tb = gops * 4;  // 4 bytes per float
        printf("  %-30s %.2f ms  %5.0f Greads/s  %5.0f GB/s\n",
               name, best, gops, tb);
    };

    printf("# B300 constant memory broadcast vs SHMEM (1M reads/thread)\n\n");
    bench([&]{ cmem_uniform_read<<<blocks, threads>>>(d_out, iters); }, "cmem uniform (broadcast)");
    bench([&]{ cmem_diverge_read<<<blocks, threads>>>(d_out, iters, 1); }, "cmem diverge (per-thread)");
    bench([&]{ smem_uniform_read<<<blocks, threads>>>(d_out, iters); }, "smem uniform");

    return 0;
}
