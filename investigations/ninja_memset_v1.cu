// NINJA: explore the absolute upper bound of HBM write throughput
// Hypothesis: v8 32-iter recipe leaves perf on table due to per-warp tail
// effects. Smaller per-warp work = more warps in flight = better saturation.

#include <cuda_runtime.h>
#include <cstdio>

template <int IT, int VW>
__launch_bounds__(256, 8) __global__ void w_template(int *data, int v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *warp_base = data + warp_id * (IT * 32 * VW);
    #pragma unroll
    for (int it = 0; it < IT; it++) {
        int *p = warp_base + (it * 32 + lane) * VW;
        if constexpr (VW == 8) {
            asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
                :: "l"(p), "r"(v) : "memory");
        } else if constexpr (VW == 4) {
            asm volatile("st.global.v4.b32 [%0], {%1,%1,%1,%1};"
                :: "l"(p), "r"(v) : "memory");
        } else {
            asm volatile("st.global.b32 [%0], %1;"
                :: "l"(p), "r"(v) : "memory");
        }
    }
}

template <int IT, int VW>
double bench(int *d_data, size_t bytes, cudaEvent_t e0, cudaEvent_t e1) {
    int per_warp_bytes = IT * 32 * VW * 4;
    int blocks_per_buf = bytes / (256 * per_warp_bytes / 32);
    if (blocks_per_buf < 1) return 0;
    int threads = 256;

    for (int i = 0; i < 3; i++) w_template<IT, VW><<<blocks_per_buf, threads>>>(d_data, 0xab);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        w_template<IT, VW><<<blocks_per_buf, threads>>>(d_data, 0xab);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    return bytes / (best/1000) / 1e9;
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;
    int *d_data; cudaMalloc(&d_data, bytes);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("# HBM write peak hunt — vary per-warp burst size\n");
    printf("# Format: IT=iters per warp, VW=vector width (4=v4, 8=v8)\n");
    printf("# Per-warp burst size = IT * 32 * VW * 4 bytes\n");
    printf("# IT  VW  per-warp(B)  GB/s    %% of 7672 spec\n");

    #define ROW(IT, VW) do { \
        double gbs = bench<IT, VW>(d_data, bytes, e0, e1); \
        printf("  %3d  %d  %8d    %6.1f  %.1f%%\n", IT, VW, IT*32*VW*4, gbs, gbs/7672*100); \
    } while(0)

    // v8 sweep
    ROW(1, 8);   // 1 KB per warp
    ROW(2, 8);   // 2 KB
    ROW(4, 8);   // 4 KB
    ROW(8, 8);   // 8 KB
    ROW(16, 8);  // 16 KB
    ROW(32, 8);  // 32 KB (current recipe)
    ROW(64, 8);  // 64 KB
    ROW(128, 8); // 128 KB

    // v4 sweep
    ROW(1, 4);
    ROW(2, 4);
    ROW(4, 4);
    ROW(8, 4);
    ROW(16, 4);
    ROW(32, 4);

    return 0;
}
