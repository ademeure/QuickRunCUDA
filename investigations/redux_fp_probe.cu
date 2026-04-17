// Try redux.sync FP variants
#include <cuda_runtime.h>
#include <cstdio>

__global__ void try_f32_min(float *out, int iters) {
    float a = threadIdx.x * 0.1f + 1.0f;
    for (int i = 0; i < iters; i++) {
        float r;
        // Try FP min reduction
        asm volatile("redux.sync.min.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(a));
        a = r + i;
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

__global__ void try_f32_add(float *out, int iters) {
    float a = threadIdx.x * 0.1f + 1.0f;
    for (int i = 0; i < iters; i++) {
        float r;
        asm volatile("redux.sync.add.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(a));
        a = r + i;
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * sizeof(float));

    int iters = 1000;
    int blocks = 1, threads = 32;

    printf("# B300 redux.sync FP variants probe\n\n");

    try_f32_min<<<blocks, threads>>>(d_out, iters);
    cudaError_t err = cudaDeviceSynchronize();
    printf("  redux.sync.min.f32: %s\n", cudaGetErrorString(err));
    cudaGetLastError();

    try_f32_add<<<blocks, threads>>>(d_out, iters);
    err = cudaDeviceSynchronize();
    printf("  redux.sync.add.f32: %s\n", cudaGetErrorString(err));

    return 0;
}
