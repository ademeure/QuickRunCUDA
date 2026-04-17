// PTX cache hint loads: __ldca, __ldcg, __ldcs, __ldlu
#include <cuda_runtime.h>
#include <cstdio>

template<typename Op>
__global__ void cached_read(const int *data, int *out, int N, int iters, Op op) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int a = 0;
    for (int i = 0; i < iters; i++) {
        for (int j = tid; j < N; j += stride) a += op(data + j);
    }
    if (a < -1e30f) out[tid] = a;
}

int main() {
    cudaSetDevice(0);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148, threads = 256;
    int *d_out; cudaMalloc(&d_out, blocks * threads * sizeof(int));

    auto bench = [&](auto launch) {
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
        return best;
    };

    int sizes_mb[] = {16, 256, 1024};
    int iters_per[] = {100, 10, 3};
    const char *fits[] = {"L2", "DRAM", "DRAM"};

    printf("# B300 cache-hint load throughput\n");
    printf("# %-10s %-10s %-12s %-12s %-12s %-12s %-12s\n",
           "size", "fits", "default", "ldca", "ldcg", "ldcs", "ldlu");

    for (int s = 0; s < 3; s++) {
        size_t bytes = (size_t)sizes_mb[s] * 1024 * 1024;
        int N = bytes / 4;
        int iters = iters_per[s];

        int *d_data; cudaMalloc(&d_data, bytes);
        cudaMemset(d_data, 0, bytes);

        float t_def = bench([&]{ cached_read<<<blocks, threads>>>(d_data, d_out, N, iters,
            [] __device__ (const int *p) { return *p; }); });
        float t_ca = bench([&]{ cached_read<<<blocks, threads>>>(d_data, d_out, N, iters,
            [] __device__ (const int *p) { int v; asm("ld.global.ca.s32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }); });
        float t_cg = bench([&]{ cached_read<<<blocks, threads>>>(d_data, d_out, N, iters,
            [] __device__ (const int *p) { int v; asm("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }); });
        float t_cs = bench([&]{ cached_read<<<blocks, threads>>>(d_data, d_out, N, iters,
            [] __device__ (const int *p) { int v; asm("ld.global.cs.s32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }); });
        float t_lu = bench([&]{ cached_read<<<blocks, threads>>>(d_data, d_out, N, iters,
            [] __device__ (const int *p) { int v; asm("ld.global.lu.s32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }); });

        double total_bytes = (double)N * iters * 4;
        printf("  %-10d %-10s %-12.0f %-12.0f %-12.0f %-12.0f %-12.0f\n",
               sizes_mb[s], fits[s],
               total_bytes/(t_def/1000)/1e9, total_bytes/(t_ca/1000)/1e9,
               total_bytes/(t_cg/1000)/1e9, total_bytes/(t_cs/1000)/1e9,
               total_bytes/(t_lu/1000)/1e9);

        cudaFree(d_data);
    }

    return 0;
}
