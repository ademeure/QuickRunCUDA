// Probe: what kernel does cudaMemset actually launch?
#include <cuda_runtime.h>
#include <cstdio>
int main() {
    cudaSetDevice(0);
    void *d; cudaMalloc(&d, 4ull * 1024 * 1024 * 1024);
    cudaMemset(d, 0xab, 4ull * 1024 * 1024 * 1024);
    cudaDeviceSynchronize();
    return 0;
}
