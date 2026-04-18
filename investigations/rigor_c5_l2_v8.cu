// C5 RIGOR: L2 BW true peak with v8 + 8-ILP recipe.
// THEORETICAL: catalog claim is L2 ~17 TB/s peak, ~23 TB/s for 64 MB workload.
// Try v8 + per-warp coalesced + ILP=8 — same recipe that lifted HBM 6→7.3.

#include <cuda_runtime.h>
#include <cstdio>

#ifndef WORK_MB
#define WORK_MB 64
#endif

extern "C" __launch_bounds__(256, 8) __global__ void l2_read_v8(int *data, int *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *warp_base = data + warp_id * (32 * 1024 / 4);  // each warp gets 32 KB region
    int acc = 0;
    #pragma unroll 1
    for (int it = 0; it < iters; it++) {
        // 8 v8 loads = 8 × 32 B = 256 B per lane, 32 lanes × 256 B = 8 KB per warp
        // (with iter-dependent shift to defeat caching)
        int shift = (it * 7) & 7;
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            int *p = warp_base + ((u + shift) * 32 + lane) * 8;
            int r0,r1,r2,r3,r4,r5,r6,r7;
            asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
                : "l"(p));
            acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
        }
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
    else if (tid == 0) out[0] = acc;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("# L2 = %.1f MB, working set = %d MB\n", prop.l2CacheSize/1e6, WORK_MB);

    size_t bytes = (size_t)WORK_MB * 1024 * 1024;
    // each warp owns 32 KB region; total warps = bytes / 32 KB
    long warps = bytes / (32 * 1024);
    int threads = 256;
    int blocks = warps / 8;

    int *d_data; cudaMalloc(&d_data, bytes); cudaMemset(d_data, 0xab, bytes);
    int *d_out; cudaMalloc(&d_out, blocks * threads * sizeof(int));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 64;
    for (int i = 0; i < 5; i++) l2_read_v8<<<blocks, threads>>>(d_data, d_out, iters);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        l2_read_v8<<<blocks, threads>>>(d_data, d_out, iters);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    // Per warp per iter: 8 × 8 × 4 B = 256 B per lane × 32 lanes = 8 KB
    long bytes_per_kernel = warps * iters * 8 * 1024;
    double tbs = bytes_per_kernel / (best/1000) / 1e12;
    printf("  WORK=%d MB blocks=%d iters=%d: %.4f ms = %.2f TB/s\n",
           WORK_MB, blocks, iters, best, tbs);
    return 0;
}
