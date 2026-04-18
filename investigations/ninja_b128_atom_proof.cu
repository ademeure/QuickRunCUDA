// Use .b128 register type explicitly
#include <cuda_runtime.h>
#include <cstdio>
extern "C" __global__ void try_b128_exch(unsigned int *p) {
    if (threadIdx.x != 0) return;
    unsigned int v0 = 1, v1 = 2, v2 = 3, v3 = 4;
    unsigned int r0, r1, r2, r3;
    asm volatile(
        "{\n"
        ".reg .b128 d, b;\n"
        "mov.b128 b, {%4, %5, %6, %7};\n"
        "atom.global.b128.exch d, [%8], b;\n"
        "mov.b128 {%0, %1, %2, %3}, d;\n"
        "}\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(v0), "r"(v1), "r"(v2), "r"(v3), "l"(p)
        : "memory"
    );
    p[0] = r0;
}

int main() {
    cudaSetDevice(0);
    unsigned int *d_p; cudaMalloc(&d_p, 256);
    cudaMemset(d_p, 0xab, 256);
    try_b128_exch<<<1, 32>>>(d_p);
    cudaError_t e = cudaDeviceSynchronize();
    printf("result: %s\n", cudaGetErrorString(e));
    return 0;
}
