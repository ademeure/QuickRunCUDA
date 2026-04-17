// __device__ symbol vs __constant__ vs global pointer access patterns
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__device__ float dev_var = 3.14f;
__device__ float dev_arr[1024];
__constant__ float c_var = 3.14f;
__constant__ float c_arr[1024];

extern "C" __global__ void access_dev_var(float *out, int iters) {
    float a = 0;
    for (int i = 0; i < iters; i++) a += dev_var;
    if (a < -1e30f) out[blockIdx.x] = a;
}

extern "C" __global__ void access_c_var(float *out, int iters) {
    float a = 0;
    for (int i = 0; i < iters; i++) a += c_var;
    if (a < -1e30f) out[blockIdx.x] = a;
}

extern "C" __global__ void access_dev_arr(float *out, int iters) {
    float a = 0;
    int tid = threadIdx.x;
    for (int i = 0; i < iters; i++) a += dev_arr[(tid + i) & 1023];
    if (a < -1e30f) out[blockIdx.x] = a;
}

extern "C" __global__ void access_c_arr(float *out, int iters) {
    float a = 0;
    int tid = threadIdx.x;
    for (int i = 0; i < iters; i++) a += c_arr[(tid + i) & 1023];
    if (a < -1e30f) out[blockIdx.x] = a;
}

extern "C" __global__ void access_global_ptr(const float *p, float *out, int iters) {
    float a = 0;
    int tid = threadIdx.x;
    for (int i = 0; i < iters; i++) a += p[(tid + i) & 1023];
    if (a < -1e30f) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * sizeof(float));

    // Init arrays
    float vals[1024];
    for (int i = 0; i < 1024; i++) vals[i] = (float)i;
    cudaMemcpyToSymbol(dev_arr, vals, sizeof(vals));
    cudaMemcpyToSymbol(c_arr, vals, sizeof(vals));

    float *d_global; cudaMalloc(&d_global, sizeof(vals));
    cudaMemcpy(d_global, vals, sizeof(vals), cudaMemcpyHostToDevice);

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
        long total_reads = (long)blocks * threads * iters;
        double gops = total_reads / (best/1000.0) / 1e9;
        printf("  %-30s %.3f ms  %.0f Greads/s\n", name, best, gops);
    };

    printf("# B300 symbol/constant/global access patterns\n");
    printf("# 148 × 128 thr × 100k iter\n\n");

    printf("## Scalar (uniform) access:\n");
    bench([&]{ access_dev_var<<<blocks, threads>>>(d_out, iters); }, "dev scalar");
    bench([&]{ access_c_var<<<blocks, threads>>>(d_out, iters); }, "constant scalar");

    printf("\n## Per-thread divergent array index:\n");
    bench([&]{ access_dev_arr<<<blocks, threads>>>(d_out, iters); }, "dev array");
    bench([&]{ access_c_arr<<<blocks, threads>>>(d_out, iters); }, "constant array");
    bench([&]{ access_global_ptr<<<blocks, threads>>>(d_global, d_out, iters); }, "global pointer");

    return 0;
}
