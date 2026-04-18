#include <cuda_runtime.h>
#include <cstdio>

// Single kernel doing both R and W simultaneously per thread
// 1024 B read + 1024 B write per thread, per-warp coalesced
__launch_bounds__(256, 8) __global__ void rw_v8(int *src, int *dst, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *src_base = src + warp_id * (32 * 1024 / 4);
    int *dst_base = dst + warp_id * (32 * 1024 / 4);
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *rp = src_base + (it * 32 + lane) * 8;
        int *wp = dst_base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(rp));
        acc ^= r0 ^ r1 ^ r2 ^ r3;
        asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
            :: "l"(wp), "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7) : "memory");
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4096ul * 1024 * 1024;
    int *d_a; cudaMalloc(&d_a, bytes); cudaMemset(d_a, 0xab, bytes);
    int *d_b; cudaMalloc(&d_b, bytes);
    int *d_out; cudaMalloc(&d_out, 16384 * 256 * sizeof(int));
    int blocks = bytes / (256 * 1024), threads = 256;

    for (int i = 0; i < 10; i++) rw_v8<<<blocks, threads>>>(d_a, d_b, d_out);
    cudaDeviceSynchronize();
    return 0;
}
