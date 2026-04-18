// HBM read SoL via v8 + per-warp coalesced + non-persistent
#include <cuda_runtime.h>
#include <cstdio>

// Each thread reads 1024 B (32 × v8 = 32 × 32 B)
// Per-warp coalesced: 32 lanes × 32 B = 1024 B contiguous per iter
__global__ void r_v8_coalesced(int *data, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int *warp_base = data + warp_id * (32 * 1024 / 4);

    int acc0=0, acc1=0, acc2=0, acc3=0, acc4=0, acc5=0, acc6=0, acc7=0;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = warp_base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(p));
        acc0 ^= r0; acc1 ^= r1; acc2 ^= r2; acc3 ^= r3;
        acc4 ^= r4; acc5 ^= r5; acc6 ^= r6; acc7 ^= r7;
    }
    int s = acc0 ^ acc1 ^ acc2 ^ acc3 ^ acc4 ^ acc5 ^ acc6 ^ acc7;
    if (s == 0xdeadbeef) out[tid] = s;
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    size_t bytes = 4096ul * 1024 * 1024;
    int *d; cudaMalloc(&d, bytes);
    cudaMemset(d, 0xab, bytes);
    int *d_out; cudaMalloc(&d_out, 16384 * 256 * sizeof(int));

    int threads = 256;
    int n_warps = bytes / (32 * 1024);
    int blocks = n_warps / 8;
    printf("# v8 per-warp coalesced READ: %d blocks × %d thr (%d warps)\n",
           blocks, threads, n_warps);

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

    float t = bench([&]{ r_v8_coalesced<<<blocks, threads>>>(d, d_out); });
    double bw = bytes/(t/1000)/1e9;
    printf("# %-30s %-12s %-12s\n", "method", "ms", "GB/s");
    printf("  %-30s %.3f ms  %.0f GB/s  (%.1f%% of 7672)\n",
           "v8 read per-warp coalesced", t, bw, bw/7672*100);
    return 0;
}
