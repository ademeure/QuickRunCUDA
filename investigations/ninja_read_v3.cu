// HBM READ ninja v3: try varying threads/block for max parallelism
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void r_2k(const int *data, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    const int *base = data + warp_id * (2 * 32 * 8);  // 2 KB per warp
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < 2; it++) {
        const int *p = base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(p));
        acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
    else if (tid == 0) out[0] = acc;
}

// Try ld.global.cs for streaming hint (might affect L2 caching)
__launch_bounds__(256, 8) __global__ void r_2k_cs(const int *data, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    const int *base = data + warp_id * (2 * 32 * 8);
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < 2; it++) {
        const int *p = base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.cs.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(p));
        acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
    else if (tid == 0) out[0] = acc;
}

// .lu = last-use hint; .ca = cache-all
__launch_bounds__(256, 8) __global__ void r_2k_lu(const int *data, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    const int *base = data + warp_id * (2 * 32 * 8);
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < 2; it++) {
        const int *p = base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.lu.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(p));
        acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
    else if (tid == 0) out[0] = acc;
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;
    int *d_data; cudaMalloc(&d_data, bytes); cudaMemset(d_data, 0xab, bytes);
    int *d_out; cudaMalloc(&d_out, 1<<20);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int blocks = bytes / (256 * 64);  // 64 B per thread = 256K blocks

    auto bench = [&](auto launch, const char* label) {
        for (int i = 0; i < 5; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double gbs = bytes / (best/1000) / 1e9;
        printf("  %s: %.4f ms = %.1f GB/s = %.2f%% spec\n",
               label, best, gbs, gbs/7672*100);
    };

    bench([&]{ r_2k<<<blocks, 256>>>(d_data, d_out); }, "default ld.global  ");
    bench([&]{ r_2k_cs<<<blocks, 256>>>(d_data, d_out); }, "ld.global.cs       ");
    bench([&]{ r_2k_lu<<<blocks, 256>>>(d_data, d_out); }, "ld.global.lu       ");
    return 0;
}
