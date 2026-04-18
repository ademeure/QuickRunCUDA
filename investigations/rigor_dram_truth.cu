#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__launch_bounds__(256, 8) __global__ void w_user(int4 *data, int N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 v = make_int4(seed, seed+1, seed+2, seed+3);
    for (int i = tid; i < N - 7*stride; i += 8*stride) {
        data[i] = v; data[i+stride] = v; data[i+2*stride] = v; data[i+3*stride] = v;
        data[i+4*stride] = v; data[i+5*stride] = v; data[i+6*stride] = v; data[i+7*stride] = v;
    }
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    size_t bytes = 4096ul * 1024 * 1024;
    int N = bytes / 16;
    int4 *d; cudaMalloc(&d, bytes);

    // First: warm up + measure user kernel
    for (int i = 0; i < 3; i++) w_user<<<148, 256>>>(d, N, 1);
    cudaDeviceSynchronize();

    float best_user = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        w_user<<<148, 256>>>(d, N, 1);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_user) best_user = ms;
    }

    // cudaMemset
    for (int i = 0; i < 3; i++) cudaMemsetAsync(d, 0xab, bytes, 0);
    cudaDeviceSynchronize();

    float best_memset = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        cudaMemsetAsync(d, 0xab, bytes, 0);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_memset) best_memset = ms;
    }

    printf("# Wall-clock effective BW for 4 GB writes:\n");
    printf("  User kernel STG.E.128: %.3f ms = %.0f GB/s\n", best_user, bytes/(best_user/1000)/1e9);
    printf("  cudaMemset:            %.3f ms = %.0f GB/s\n", best_memset, bytes/(best_memset/1000)/1e9);

    return 0;
}
