// Constant memory bandwidth on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CMEM_SIZE 16384  // max const memory in CUDA = 64 KB
__constant__ float c_data[CMEM_SIZE];  // 64 KB

extern "C" __global__ void cmem_uniform(float *out, int N, int reps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0;
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < N; i++) {
            acc += c_data[i];  // ALL threads read same address (broadcast)
        }
    }
    if (acc == -42.0f) out[tid] = acc;
}

extern "C" __global__ void cmem_divergent(float *out, int N, int reps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0;
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < N; i++) {
            int idx = (tid + i) & (CMEM_SIZE - 1);
            acc += c_data[idx];  // each thread reads different address
        }
    }
    if (acc == -42.0f) out[tid] = acc;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    printf("# B300 constant memory test\n");
    printf("# Total const memory: %zu\n\n", prop.totalConstMem);

    // Init constant memory
    float h_data[CMEM_SIZE];
    for (int i = 0; i < CMEM_SIZE; i++) h_data[i] = 1.0f + i * 0.001f;
    cudaMemcpyToSymbol(c_data, h_data, sizeof(h_data));

    float *d_out;
    cudaMalloc(&d_out, prop.multiProcessorCount * 256 * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    int blocks = prop.multiProcessorCount, threads = 128;
    int reps = 100;

    int N_arr[] = {16, 64, 256, 1024, 4096, 16384};

    printf("## Uniform access (all threads same address - broadcast)\n");
    printf("# %-10s %-12s\n", "N_floats", "BW_GB/s");
    for (int N : N_arr) {
        float t = bench([&]{
            cmem_uniform<<<blocks, threads, 0, s>>>(d_out, N, reps);
        });
        size_t total_bytes = (size_t)blocks * threads * N * reps * 4;
        printf("  %-10d %-12.1f\n", N, total_bytes / (t/1e3) / 1e9);
    }

    printf("\n## Divergent access (each thread different address)\n");
    for (int N : N_arr) {
        float t = bench([&]{
            cmem_divergent<<<blocks, threads, 0, s>>>(d_out, N, reps);
        });
        size_t total_bytes = (size_t)blocks * threads * N * reps * 4;
        printf("  %-10d %-12.1f\n", N, total_bytes / (t/1e3) / 1e9);
    }

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
