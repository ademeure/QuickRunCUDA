// V6 K4: Kernel parameter passing cost
// Test launch overhead with various param sizes:
//   small (4 args): 4 ptrs
//   medium (16 args): 16 ints
//   large (256 args via struct): 256 ints in struct
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

struct ParamsBig {
    int data[256];  // 1 KB
};

__global__ void kernel_small(unsigned int* a, unsigned int* b, unsigned int* c, unsigned int* d) {
    if (threadIdx.x == 0 && blockIdx.x == 0) a[0] = 1;
}

__global__ void kernel_medium(int x0, int x1, int x2, int x3, int x4, int x5, int x6, int x7,
                              int x8, int x9, int x10, int x11, int x12, int x13, int x14, int x15,
                              unsigned int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = x0 + x15;
}

__global__ void kernel_big(ParamsBig p, unsigned int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = p.data[0] + p.data[255];
}

int main() {
    cudaSetDevice(0);
    unsigned int *a, *b, *c, *d, *out;
    cudaMalloc(&a, 4); cudaMalloc(&b, 4); cudaMalloc(&c, 4); cudaMalloc(&d, 4);
    cudaMallocManaged(&out, 4);

    int N_RUNS = 5000;

    // Warmup each kernel
    kernel_small<<<1, 32>>>(a, b, c, d);
    kernel_medium<<<1, 32>>>(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, out);
    ParamsBig p = {};
    kernel_big<<<1, 32>>>(p, out);
    cudaDeviceSynchronize();

    // Test 1: small
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        kernel_small<<<1, 32>>>(a, b, c, d);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double small_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    // Test 2: medium (16 ints + 1 ptr)
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        kernel_medium<<<1, 32>>>(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, out);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double medium_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;

    // Test 3: big (1 KB struct)
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        kernel_big<<<1, 32>>>(p, out);
    }
    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();
    double big_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / N_RUNS;

    printf("Kernel launch with:\n");
    printf("  4 ptr args (32 B):     %.2f us\n", small_us);
    printf("  16 ints + 1 ptr (72 B):%.2f us (%.2fx)\n", medium_us, medium_us/small_us);
    printf("  1 KB struct + 1 ptr:   %.2f us (%.2fx)\n", big_us, big_us/small_us);

    return 0;
}
