// Test write-allocate hypothesis for HBM write gap
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

// Default writes (cache-allocate)
__launch_bounds__(512, 4) __global__ void write_default(int4 *data, int N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 v0 = make_int4(seed, seed+1, seed+2, seed+3);
    int4 v1 = make_int4(seed*2, seed*2+1, seed*2+2, seed*2+3);
    int4 v2 = make_int4(seed*3, seed*3+1, seed*3+2, seed*3+3);
    int4 v3 = make_int4(seed*4, seed*4+1, seed*4+2, seed*4+3);
    int4 v4 = make_int4(seed*5, seed*5+1, seed*5+2, seed*5+3);
    int4 v5 = make_int4(seed*6, seed*6+1, seed*6+2, seed*6+3);
    int4 v6 = make_int4(seed*7, seed*7+1, seed*7+2, seed*7+3);
    int4 v7 = make_int4(seed*8, seed*8+1, seed*8+2, seed*8+3);
    for (int i = tid; i < N - 7*stride; i += 8*stride) {
        data[i] = v0;
        data[i + stride] = v1;
        data[i + 2*stride] = v2;
        data[i + 3*stride] = v3;
        data[i + 4*stride] = v4;
        data[i + 5*stride] = v5;
        data[i + 6*stride] = v6;
        data[i + 7*stride] = v7;
    }
}

// Cache-streaming writes via st.global.cs PTX hint
__launch_bounds__(512, 4) __global__ void write_cs(int4 *data, int N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 v0 = make_int4(seed, seed+1, seed+2, seed+3);
    int4 v1 = make_int4(seed*2, seed*2+1, seed*2+2, seed*2+3);
    int4 v2 = make_int4(seed*3, seed*3+1, seed*3+2, seed*3+3);
    int4 v3 = make_int4(seed*4, seed*4+1, seed*4+2, seed*4+3);
    for (int i = tid; i < N - 3*stride; i += 4*stride) {
        asm volatile("st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};" :: "l"(&data[i]),            "r"(v0.x),"r"(v0.y),"r"(v0.z),"r"(v0.w) : "memory");
        asm volatile("st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};" :: "l"(&data[i+stride]),     "r"(v1.x),"r"(v1.y),"r"(v1.z),"r"(v1.w) : "memory");
        asm volatile("st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};" :: "l"(&data[i+2*stride]),   "r"(v2.x),"r"(v2.y),"r"(v2.z),"r"(v2.w) : "memory");
        asm volatile("st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};" :: "l"(&data[i+3*stride]),   "r"(v3.x),"r"(v3.y),"r"(v3.z),"r"(v3.w) : "memory");
    }
}

// Write-back hint (.wb) — explicit allocate
__launch_bounds__(512, 4) __global__ void write_wb(int4 *data, int N, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int4 v0 = make_int4(seed, seed+1, seed+2, seed+3);
    int4 v1 = make_int4(seed*2, seed*2+1, seed*2+2, seed*2+3);
    int4 v2 = make_int4(seed*3, seed*3+1, seed*3+2, seed*3+3);
    int4 v3 = make_int4(seed*4, seed*4+1, seed*4+2, seed*4+3);
    for (int i = tid; i < N - 3*stride; i += 4*stride) {
        asm volatile("st.global.wb.v4.s32 [%0], {%1,%2,%3,%4};" :: "l"(&data[i]),            "r"(v0.x),"r"(v0.y),"r"(v0.z),"r"(v0.w) : "memory");
        asm volatile("st.global.wb.v4.s32 [%0], {%1,%2,%3,%4};" :: "l"(&data[i+stride]),     "r"(v1.x),"r"(v1.y),"r"(v1.z),"r"(v1.w) : "memory");
        asm volatile("st.global.wb.v4.s32 [%0], {%1,%2,%3,%4};" :: "l"(&data[i+2*stride]),   "r"(v2.x),"r"(v2.y),"r"(v2.z),"r"(v2.w) : "memory");
        asm volatile("st.global.wb.v4.s32 [%0], {%1,%2,%3,%4};" :: "l"(&data[i+3*stride]),   "r"(v3.x),"r"(v3.y),"r"(v3.z),"r"(v3.w) : "memory");
    }
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    size_t bytes = 4096ul * 1024 * 1024;
    int N = bytes / 16;
    int4 *d; cudaMalloc(&d, bytes);

    auto bench = [&](auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 7; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# Write-allocate audit: 4 GB int4 writes via various PTX hints\n\n");
    printf("# %-25s %-12s %-12s %-12s\n", "method", "ms", "GB/s", "%peak");

    {
        float t = bench([&]{ write_default<<<148, 512>>>(d, N, 1); });
        double bw = bytes/(t/1000)/1e9;
        printf("  %-25s %-12.3f %-12.0f %-12.1f\n", "default ST.E", t, bw, bw/7672*100);
    }
    {
        float t = bench([&]{ write_cs<<<148, 512>>>(d, N, 1); });
        double bw = bytes/(t/1000)/1e9;
        printf("  %-25s %-12.3f %-12.0f %-12.1f\n", "st.global.cs (no-alloc)", t, bw, bw/7672*100);
    }
    {
        float t = bench([&]{ write_wb<<<148, 512>>>(d, N, 1); });
        double bw = bytes/(t/1000)/1e9;
        printf("  %-25s %-12.3f %-12.0f %-12.1f\n", "st.global.wb (allocate)", t, bw, bw/7672*100);
    }
    {
        float t = bench([&]{ cudaMemsetAsync(d, 0xab, bytes, 0); });
        double bw = bytes/(t/1000)/1e9;
        printf("  %-25s %-12.3f %-12.0f %-12.1f\n", "cudaMemset (reference)", t, bw, bw/7672*100);
    }

    return 0;
}
