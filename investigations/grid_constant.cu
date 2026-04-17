// __grid_constant__ kernel parameters (CUDA 11.7+)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

struct BigArgs {
    int values[256];  // 1 KB struct
};

__global__ void with_grid_const(__grid_constant__ const BigArgs args, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = args.values[tid & 255];
}

__global__ void without_grid_const(BigArgs args, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = args.values[tid & 255];
}

__global__ void via_pointer(const BigArgs *args, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    out[tid] = args->values[tid & 255];
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));

    BigArgs args;
    for (int i = 0; i < 256; i++) args.values[i] = i;

    BigArgs *d_args;
    cudaMalloc(&d_args, sizeof(BigArgs));
    cudaMemcpy(d_args, &args, sizeof(BigArgs), cudaMemcpyHostToDevice);

    auto bench = [&](auto fn, int trials = 200) {
        for (int i = 0; i < 5; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 __grid_constant__ vs by-value vs pointer kernel args\n");
    printf("# 1 KB struct passed to kernel\n\n");

    {
        float t = bench([&]{
            with_grid_const<<<1, 256, 0, s>>>(args, d_out);
        });
        printf("  __grid_constant__ struct:      %.2f us\n", t);
    }
    {
        float t = bench([&]{
            without_grid_const<<<1, 256, 0, s>>>(args, d_out);
        });
        printf("  plain by-value struct:         %.2f us\n", t);
    }
    {
        float t = bench([&]{
            via_pointer<<<1, 256, 0, s>>>(d_args, d_out);
        });
        printf("  via pointer (post-memcpy):     %.2f us\n", t);
    }

    return 0;
}
