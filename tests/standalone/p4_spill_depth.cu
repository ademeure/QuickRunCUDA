// P4: Max register spill depth
// Spilling on B300 = LMEM allocation = scales to GB/thread (per P5)
// Test: huge stack arrays force massive spilling
#include <cstdio>
#include <cuda_runtime.h>

template<int N>
__global__ __launch_bounds__(32, 1) void k(float* out, int u2) {
    float vars[N];
    for (int i = 0; i < N; i++) vars[i] = (float)threadIdx.x + (float)i + (float)u2 * 1e-9f;
    float sum = 0;
    for (int i = 0; i < N; i++) sum += vars[(i * 137 + u2) % N];
    if ((int)sum == u2) out[0] = sum;
}

template<int N>
void test() {
    float* d; cudaMalloc(&d, 4);
    k<N><<<1, 32>>>(d, 7);
    cudaError_t err = cudaDeviceSynchronize();
    printf("N=%d: %s\n", N, err == cudaSuccess ? "OK" : cudaGetErrorString(err));
    cudaFree(d);
}

int main() {
    test<100>();
    test<1000>();
    test<10000>();     // 40 KB/thread
    test<100000>();    // 400 KB/thread
    test<1000000>();   // 4 MB/thread
    test<10000000>();  // 40 MB/thread (1.3 GB total)
    return 0;
}
