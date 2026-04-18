// L1 axpy v2: try cache-hint variants + 16-elem ILP + persistent
// Theoretical:
//   HBM read peak 7.31 TB/s; A6 finding: mixed 2R:1W achievable ~7.0 TB/s
//   Per element: 2R + 1W BF16 = 6 bytes
//   SoL Gelems/s = 7.0e12 / 6 = 1.167 Gelems/s
//   For 1 GB y_bytes (512M elems): SoL time = 0.439 ms
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Variant 1: original (baseline reproducer)
__launch_bounds__(256, 4) __global__ void axpy_v1(
    const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y, float a, int N)
{
    int stride = gridDim.x * blockDim.x;
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 8;
    int lane = threadIdx.x & 31;
    __nv_bfloat16 a_bf = __float2bfloat16(a);
    for (int i = warp_base + lane * 8; i < N - 7; i += stride * 8) {
        uint4 xv = *(uint4*)&x[i];
        uint4 yv = *(uint4*)&y[i];
        const __nv_bfloat16 *xb = (const __nv_bfloat16*)&xv;
        __nv_bfloat16 *yb = (__nv_bfloat16*)&yv;
        #pragma unroll
        for (int j = 0; j < 8; j++) yb[j] = __hadd(__hmul(xb[j], a_bf), yb[j]);
        *(uint4*)&y[i] = yv;
    }
}

// Variant 2: same as v1 but use PTX inline for STG.E.128 with .cs (streaming/no-L2-allocate)
__launch_bounds__(256, 4) __global__ void axpy_v2_cs_write(
    const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y, float a, int N)
{
    int stride = gridDim.x * blockDim.x;
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 8;
    int lane = threadIdx.x & 31;
    __nv_bfloat16 a_bf = __float2bfloat16(a);
    for (int i = warp_base + lane * 8; i < N - 7; i += stride * 8) {
        uint4 xv = *(uint4*)&x[i];
        uint4 yv = *(uint4*)&y[i];
        const __nv_bfloat16 *xb = (const __nv_bfloat16*)&xv;
        __nv_bfloat16 *yb = (__nv_bfloat16*)&yv;
        #pragma unroll
        for (int j = 0; j < 8; j++) yb[j] = __hadd(__hmul(xb[j], a_bf), yb[j]);
        // Streaming store: bypass L2 allocation
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "l"(&y[i]), "r"(yv.x), "r"(yv.y), "r"(yv.z), "r"(yv.w) : "memory");
    }
}

// Variant 3: ILP=2 — load 16 BF16 per thread (2x uint4)
__launch_bounds__(256, 4) __global__ void axpy_v3_ilp2(
    const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y, float a, int N)
{
    int stride = gridDim.x * blockDim.x;
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 16;
    int lane = threadIdx.x & 31;
    __nv_bfloat16 a_bf = __float2bfloat16(a);
    for (int i = warp_base + lane * 16; i < N - 15; i += stride * 16) {
        uint4 xv0 = *(uint4*)&x[i];
        uint4 xv1 = *(uint4*)&x[i + 8];
        uint4 yv0 = *(uint4*)&y[i];
        uint4 yv1 = *(uint4*)&y[i + 8];
        __nv_bfloat16 *xb0 = (__nv_bfloat16*)&xv0, *xb1 = (__nv_bfloat16*)&xv1;
        __nv_bfloat16 *yb0 = (__nv_bfloat16*)&yv0, *yb1 = (__nv_bfloat16*)&yv1;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            yb0[j] = __hadd(__hmul(xb0[j], a_bf), yb0[j]);
            yb1[j] = __hadd(__hmul(xb1[j], a_bf), yb1[j]);
        }
        *(uint4*)&y[i] = yv0;
        *(uint4*)&y[i + 8] = yv1;
    }
}

// Variant 4: ILP=2 + cs writes
__launch_bounds__(256, 4) __global__ void axpy_v4_ilp2_cs(
    const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y, float a, int N)
{
    int stride = gridDim.x * blockDim.x;
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 16;
    int lane = threadIdx.x & 31;
    __nv_bfloat16 a_bf = __float2bfloat16(a);
    for (int i = warp_base + lane * 16; i < N - 15; i += stride * 16) {
        uint4 xv0 = *(uint4*)&x[i];
        uint4 xv1 = *(uint4*)&x[i + 8];
        uint4 yv0 = *(uint4*)&y[i];
        uint4 yv1 = *(uint4*)&y[i + 8];
        __nv_bfloat16 *xb0 = (__nv_bfloat16*)&xv0, *xb1 = (__nv_bfloat16*)&xv1;
        __nv_bfloat16 *yb0 = (__nv_bfloat16*)&yv0, *yb1 = (__nv_bfloat16*)&yv1;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            yb0[j] = __hadd(__hmul(xb0[j], a_bf), yb0[j]);
            yb1[j] = __hadd(__hmul(xb1[j], a_bf), yb1[j]);
        }
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "l"(&y[i]), "r"(yv0.x), "r"(yv0.y), "r"(yv0.z), "r"(yv0.w) : "memory");
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "l"(&y[i+8]), "r"(yv1.x), "r"(yv1.y), "r"(yv1.z), "r"(yv1.w) : "memory");
    }
}

// Variant 5: ILP=4 - 32 BF16/thread
__launch_bounds__(256, 4) __global__ void axpy_v5_ilp4_cs(
    const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y, float a, int N)
{
    int stride = gridDim.x * blockDim.x;
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 32;
    int lane = threadIdx.x & 31;
    __nv_bfloat16 a_bf = __float2bfloat16(a);
    for (int i = warp_base + lane * 32; i < N - 31; i += stride * 32) {
        uint4 xv0 = *(uint4*)&x[i];
        uint4 xv1 = *(uint4*)&x[i + 8];
        uint4 xv2 = *(uint4*)&x[i + 16];
        uint4 xv3 = *(uint4*)&x[i + 24];
        uint4 yv0 = *(uint4*)&y[i];
        uint4 yv1 = *(uint4*)&y[i + 8];
        uint4 yv2 = *(uint4*)&y[i + 16];
        uint4 yv3 = *(uint4*)&y[i + 24];
        __nv_bfloat16 *xb[4] = {(__nv_bfloat16*)&xv0,(__nv_bfloat16*)&xv1,(__nv_bfloat16*)&xv2,(__nv_bfloat16*)&xv3};
        __nv_bfloat16 *yb[4] = {(__nv_bfloat16*)&yv0,(__nv_bfloat16*)&yv1,(__nv_bfloat16*)&yv2,(__nv_bfloat16*)&yv3};
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) yb[k][j] = __hadd(__hmul(xb[k][j], a_bf), yb[k][j]);
        }
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "l"(&y[i]),    "r"(yv0.x),"r"(yv0.y),"r"(yv0.z),"r"(yv0.w) : "memory");
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "l"(&y[i+8]),  "r"(yv1.x),"r"(yv1.y),"r"(yv1.z),"r"(yv1.w) : "memory");
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "l"(&y[i+16]), "r"(yv2.x),"r"(yv2.y),"r"(yv2.z),"r"(yv2.w) : "memory");
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "l"(&y[i+24]), "r"(yv3.x),"r"(yv3.y),"r"(yv3.z),"r"(yv3.w) : "memory");
    }
}

// Variant 6: persistent kernel — process whole array as grid-stride from 148*8=1184 blocks
__launch_bounds__(256, 4) __global__ void axpy_v6_persistent(
    const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y, float a, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __nv_bfloat16 a_bf = __float2bfloat16(a);
    for (int i = tid * 8; i < N - 7; i += stride * 8) {
        uint4 xv = *(uint4*)&x[i];
        uint4 yv = *(uint4*)&y[i];
        const __nv_bfloat16 *xb = (const __nv_bfloat16*)&xv;
        __nv_bfloat16 *yb = (__nv_bfloat16*)&yv;
        #pragma unroll
        for (int j = 0; j < 8; j++) yb[j] = __hadd(__hmul(xb[j], a_bf), yb[j]);
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "l"(&y[i]), "r"(yv.x), "r"(yv.y), "r"(yv.z), "r"(yv.w) : "memory");
    }
}

int main(int argc, char** argv) {
    cudaSetDevice(0);
    size_t bytes = 1ull * 1024 * 1024 * 1024;
    int N = bytes / sizeof(__nv_bfloat16);
    __nv_bfloat16 *d_x, *d_y;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    cudaMemset(d_x, 0x42, bytes); cudaMemset(d_y, 0x33, bytes);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](const char* name, void(*kfn)(const __nv_bfloat16*, __nv_bfloat16*, float, int), int blocks, int chunk) {
        // Warmup
        for (int i = 0; i < 5; i++) kfn<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) {
            printf("  ERROR in %s: %s\n", name, cudaGetErrorString(cudaGetLastError()));
            return;
        }
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        size_t total = bytes * 3;  // R(x) + R(y) + W(y)
        double gbs = total / (best/1000) / 1e9;
        printf("  %-25s blocks=%-7d %.4f ms = %.0f GB/s = %.2f%% spec (chunk=%d)\n",
               name, blocks, best, gbs, gbs/7672*100, chunk);
    };

    int blocks_optimal_v1 = 131072;
    int blocks_optimal_v3 = 65536;     // ILP=2 → half blocks
    int blocks_optimal_v5 = 32768;     // ILP=4 → quarter blocks
    int blocks_persistent = 148 * 8;   // 8 blocks/SM

    bench("v1 baseline 8/thr",         axpy_v1,            blocks_optimal_v1, 8);
    bench("v2 + cs write 8/thr",       axpy_v2_cs_write,   blocks_optimal_v1, 8);
    bench("v3 ILP=2 16/thr",           axpy_v3_ilp2,       blocks_optimal_v3, 16);
    bench("v4 ILP=2 + cs 16/thr",      axpy_v4_ilp2_cs,    blocks_optimal_v3, 16);
    bench("v5 ILP=4 + cs 32/thr",      axpy_v5_ilp4_cs,    blocks_optimal_v5, 32);
    bench("v6 persist + cs 8/thr",     axpy_v6_persistent, blocks_persistent, 8);

    // Also sweep the v4 (ILP=2 + cs) at different block counts
    printf("\n# v4 block sweep:\n");
    for (int b : {65536, 32768, 16384, 8192, 4096, 2368, 1184, 592, 148}) {
        bench("v4 ILP=2 + cs", axpy_v4_ilp2_cs, b, 16);
    }

    return 0;
}
