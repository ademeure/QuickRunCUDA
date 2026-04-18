// K2 RIGOR: STG.E.ENL2 actual semantics — does the suffix mean
// "Evict No L2" (write-through, bypass L2) or something else?
//
// Method: write workload variant -> immediately re-read. If re-read is FAST
// (L2 hit), the data was retained in L2. If re-read is SLOW (DRAM), it
// bypassed L2.

#include <cuda_runtime.h>
#include <cstdio>

#ifndef VARIANT
#define VARIANT 0
#endif

extern "C" __launch_bounds__(256, 4) __global__ void writer(int *p, int v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int *pp = p + tid * 8;
    // Use the inline asm form that produces specific cache hints
#if VARIANT == 0
    // Default st.global -> typically STG.E.128
    *(int4*)pp = make_int4(v, v, v, v);
    *((int4*)pp + 1) = make_int4(v, v, v, v);
#elif VARIANT == 1
    // st.global.cg = cache-global (only L2)
    asm volatile("st.global.cg.v4.b32 [%0], {%1,%1,%1,%1};" :: "l"(pp), "r"(v) : "memory");
    asm volatile("st.global.cg.v4.b32 [%0], {%1,%1,%1,%1};" :: "l"(pp+4), "r"(v) : "memory");
#elif VARIANT == 2
    // st.global.cs = cache-streaming (likely-not-reused; tries to bypass L2)
    asm volatile("st.global.cs.v4.b32 [%0], {%1,%1,%1,%1};" :: "l"(pp), "r"(v) : "memory");
    asm volatile("st.global.cs.v4.b32 [%0], {%1,%1,%1,%1};" :: "l"(pp+4), "r"(v) : "memory");
#elif VARIANT == 3
    // st.global.wb = cache-writeback (write-allocate); reuse expected
    asm volatile("st.global.wb.v4.b32 [%0], {%1,%1,%1,%1};" :: "l"(pp), "r"(v) : "memory");
    asm volatile("st.global.wb.v4.b32 [%0], {%1,%1,%1,%1};" :: "l"(pp+4), "r"(v) : "memory");
#elif VARIANT == 4
    // v8 default — what the recipe uses
    asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};" :: "l"(pp), "r"(v) : "memory");
#endif
}

extern "C" __launch_bounds__(256, 4) __global__ void reader(int *p, unsigned long long *clk_out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int *pp = p + tid * 8;
    unsigned long long t0 = clock64();
    int4 a = *(int4*)pp;
    int4 b = *((int4*)pp + 1);
    int sum = a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w;
    unsigned long long t1 = clock64();
    if (sum == 0xdeadbeef && tid == 0) clk_out[0] = (t1 - t0);
    else if (tid == 0) clk_out[0] = (t1 - t0);
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024;  // 4 MB — fits in L2 easily
    int *d_p; cudaMalloc(&d_p, bytes);
    unsigned long long *d_clk; cudaMalloc(&d_clk, 8);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int N = bytes / 4 / 8;  // each thread writes 8 ints
    int blocks = N / 256;

    cudaMemset(d_p, 0xff, bytes);

    // Warmup
    for (int i = 0; i < 3; i++) {
        writer<<<blocks, 256>>>(d_p, 0x42);
        reader<<<blocks, 256>>>(d_p, d_clk);
    }
    cudaDeviceSynchronize();

    // 1. Time WRITE alone
    float t_write = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        writer<<<blocks, 256>>>(d_p, 0x42);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < t_write) t_write = ms;
    }

    // 2. Time WRITE then READ (consecutive)
    // First do the write to populate cache state
    writer<<<blocks, 256>>>(d_p, 0x42);
    cudaDeviceSynchronize();

    float t_read = 1e30f;
    for (int i = 0; i < 10; i++) {
        // Re-write each iter then read
        writer<<<blocks, 256>>>(d_p, 0x42);
        cudaEventRecord(e0);
        reader<<<blocks, 256>>>(d_p, d_clk);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < t_read) t_read = ms;
    }

    unsigned long long h_clk;
    cudaMemcpy(&h_clk, d_clk, 8, cudaMemcpyDeviceToHost);

    double w_gbs = bytes / (t_write/1000) / 1e9;
    double r_gbs = bytes / (t_read/1000) / 1e9;
    printf("VARIANT %d: write %.4f ms (%.1f GB/s), read-after-write %.4f ms (%.1f GB/s), per-load %llu cy\n",
           VARIANT, t_write, w_gbs, t_read, r_gbs, h_clk);
    return 0;
}
