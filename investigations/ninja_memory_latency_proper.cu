// Proper memory latency: warm + chase in SAME kernel, no kernel-boundary flush
// Also: try multiple WS sizes spanning L1 (small) -> L2 (mid) -> DRAM (huge)
// Be careful: need to choose WS that ACTUALLY fits in L2, AND not evicted between warm and chase
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Single-thread kernel: warm cache by sequential read, then time pointer chase
// All in ONE kernel = no chance of L2 flush between warm and timing
__global__ void warm_then_chase(uint64_t *p, uint64_t N, uint64_t *out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // Warm: read all entries (places into L1 + L2)
    uint64_t acc = 0;
    for (uint64_t i = 0; i < N; i++) acc ^= p[i];
    // Force the warm to actually happen (anti-optimize)
    if (acc == 0xdeadbeefcafef00d) out[42] = acc;
    // Now time the chase
    uint64_t cur = 0;
    long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 200; i++) cur = p[cur];
    long long t1 = clock64();
    out[0] = cur;
    out[1] = (uint64_t)(t1 - t0);
    // Also: re-time IMMEDIATELY (everything still warm)
    cur = 0;
    long long t2 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 200; i++) cur = p[cur];
    long long t3 = clock64();
    out[2] = cur;
    out[3] = (uint64_t)(t3 - t2);
}

// Init: random permutation
__global__ void chase_init(uint64_t *p, uint64_t N) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    p[tid] = (tid * 65537ULL + 7919ULL) % N;
}

// Same as above but no warm phase — measures pure cold cache behavior
__global__ void cold_chase(uint64_t *p, uint64_t N, uint64_t *out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint64_t cur = 0;
    long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 200; i++) cur = p[cur];
    long long t1 = clock64();
    out[0] = cur;
    out[1] = (uint64_t)(t1 - t0);
}

int main() {
    cudaSetDevice(0);
    uint64_t Nmax = 256ULL * 1024 * 1024;
    uint64_t *d_p; cudaMalloc(&d_p, Nmax * 8);
    uint64_t *d_out; cudaMalloc(&d_out, 64);
    uint64_t h[4];

    auto run = [&](uint64_t N, const char* label) {
        chase_init<<<(N+255)/256, 256>>>(d_p, N);
        cudaDeviceSynchronize();
        // Warmup invocation
        warm_then_chase<<<1, 32>>>(d_p, N, d_out);
        cudaDeviceSynchronize();
        // Measure: warm phase happens then chase
        warm_then_chase<<<1, 32>>>(d_p, N, d_out);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d_out, 32, cudaMemcpyDeviceToHost);
        double cy_after_warm_first = (double)h[1] / 200;
        double cy_after_warm_immediate = (double)h[3] / 200;
        printf("  %-22s WS=%6llu KB  warm-then-chase: %5.1f cy = %5.1f ns  (immediate re-chase: %5.1f cy = %5.1f ns)\n",
               label, N*8/1024,
               cy_after_warm_first, cy_after_warm_first / 2.032,
               cy_after_warm_immediate, cy_after_warm_immediate / 2.032);
    };

    auto run_cold = [&](uint64_t N, const char* label) {
        chase_init<<<(N+255)/256, 256>>>(d_p, N);
        cudaDeviceSynchronize();
        // Flush the cache by touching unrelated memory (huge alloc + memset)
        // (skip — kernel boundary may evict; just do cold without warm)
        cold_chase<<<1, 32>>>(d_p, N, d_out);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d_out, 16, cudaMemcpyDeviceToHost);
        double cy = (double)h[1] / 200;
        printf("  %-22s WS=%6llu KB  cold: %5.1f cy = %5.1f ns\n",
               label, N*8/1024, cy, cy / 2.032);
    };

    printf("# Memory latency proper (single-thread chase, 200 loads, warm in same kernel)\n");
    printf("\n# L1-territory (fits in L1 ~256 KB carveout limit)\n");
    run(256, "L1 (2 KB)");           // 2 KB
    run(2 * 1024, "L1 (16 KB)");     // 16 KB
    run(8 * 1024, "L1 (64 KB)");     // 64 KB - fits L1 with carveout=0
    printf("\n# L2-territory (4 MB - 32 MB; L2 = 60 MB total)\n");
    run(64 * 1024, "L2 (512 KB)");
    run(512 * 1024, "L2 (4 MB)");
    run(2 * 1024 * 1024, "L2 (16 MB)");
    run(4 * 1024 * 1024, "L2 (32 MB)");
    printf("\n# DRAM-territory (>L2)\n");
    run(8 * 1024 * 1024, "DRAM (64 MB)");
    run(16 * 1024 * 1024, "DRAM (128 MB)");
    run(64 * 1024 * 1024, "DRAM (512 MB)");
    run(256 * 1024 * 1024, "DRAM (2 GB)");
    return 0;
}
