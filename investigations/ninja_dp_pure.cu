// DP launch without child - just call the launch with 0 work
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void child_zero() { /* nothing */ }

extern "C" __global__ void parent_pure_launches(int N) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < N; i++) {
        child_zero<<<1, 32, 0, cudaStreamFireAndForget>>>();
    }
}

extern "C" __global__ void parent_pure_atomic(int N, int *c) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < N; i++) {
        atomicAdd(c, 1);
    }
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int *d_c; cudaMalloc(&d_c, 4); cudaMemset(d_c, 0, 4);

    // Warmup
    parent_pure_launches<<<1, 32>>>(10);
    parent_pure_atomic<<<1, 32>>>(10, d_c);
    cudaDeviceSynchronize();

    for (int N : {100, 1000, 10000}) {
        cudaEventRecord(e0);
        parent_pure_launches<<<1, 32>>>(N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("DP pure launches x%d:  %.4f ms = %.2f us/launch\n", N, ms, ms*1000.0/N);
    }
    return 0;
}
