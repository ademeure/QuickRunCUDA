// V5 D5: Power-aware code patterns — cp.async vs LDG
// Same memory work via different mechanisms; measure power
#include <cuda_runtime.h>
#include <cstdio>

#ifndef MODE
#define MODE 0
#endif

__global__ __launch_bounds__(128, 4)
void kernel(unsigned int* A, int iters, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ __align__(16) unsigned int smem[2048];
    unsigned int v = (unsigned)u2;

    for (int i = 0; i < iters; i++) {
#pragma unroll 16
        for (int u = 0; u < 16; u++) {
            unsigned int* addr = A + ((i * 32 + u + gtid) & ((1 << 22) - 1));
#if MODE == 0
            // Plain LDG.E
            unsigned int x;
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(addr));
            v ^= x;
#elif MODE == 1
            // LDG.E.STRONG.GPU (.cg, bypass L1)
            unsigned int x;
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr));
            v ^= x;
#elif MODE == 2
            // cp.async to SMEM (16-byte transfer; need addr aligned)
            unsigned int* aligned = (unsigned int*)((unsigned long long)addr & ~15ull);
            unsigned int dst_off = (threadIdx.x * 16) & 1023;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         "cp.async.commit_group;\n"
                         "cp.async.wait_all;"
                :: "r"(__cvta_generic_to_shared(smem) + dst_off), "l"(aligned));
            v ^= smem[dst_off / 4];
#endif
        }
    }

    if (v == (unsigned)0xDEADBEEF) A[0] = v;
}

int main() {
    int blocks = 296, threads = 128, iters = 1000000;
    unsigned int* d;
    cudaMalloc(&d, 16ull * 1024 * 1024);
    cudaMemset(d, 0, 16ull * 1024 * 1024);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    kernel<<<blocks, threads>>>(d, 1000, 7);
    cudaDeviceSynchronize();

    cudaEventRecord(s);
    kernel<<<blocks, threads>>>(d, iters, 7);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    printf("MODE=%d time=%.3f ms\n", MODE, ms);
    return 0;
}
