// Measure both kernels during concurrent run via ncu
#include <cuda_runtime.h>
#include <cstdio>

__global__ void rw_pair(int *src, int *dst) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *src_base = src + warp_id * (32 * 1024 / 4);
    int *dst_base = dst + warp_id * (32 * 1024 / 4);
    int v = 0xab;
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *sp = src_base + (it * 32 + lane) * 8;
        int *dp = dst_base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(sp));
        acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(dp), "r"(v) : "memory");
    }
    if (acc == 0xdeadbeef) dst[tid] = acc;
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4096ul * 1024 * 1024;
    int *d_a; cudaMalloc(&d_a, bytes); cudaMemset(d_a, 0xab, bytes);
    int *d_b; cudaMalloc(&d_b, bytes);
    int blocks = bytes / (256 * 1024), threads = 256;
    for (int i = 0; i < 5; i++) rw_pair<<<blocks, threads>>>(d_a, d_b);
    cudaDeviceSynchronize();
    return 0;
}
