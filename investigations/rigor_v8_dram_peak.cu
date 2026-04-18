// User recipe with PER-WARP COALESCED v8 stores
#include <cuda_runtime.h>
#include <cstdio>

// Each thread writes 1024 B = 32 × 32 B v8 stores
// Per-warp pattern: 32 lanes × 32 B = 1024 B per warp per iter (contiguous)
// 32 iters per thread = 32 KB per warp
// Block has 8 warps = 256 KB per block
__global__ void w_v8_coalesced(int *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int v = 0xab;
    // Each warp writes 32 KB starting at warp_id * 32 KB
    // Within warp, lane writes 32 B at offset lane * 32 + iter * 32 * 32
    int *warp_base = data + warp_id * (32 * 1024 / 4);  // 32 KB / 4 bytes/int = 8192 ints
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        // Offset = (it * 32 + lane) * 32 bytes / 4 = (it*32+lane)*8 ints
        int *p = warp_base + (it * 32 + lane) * 8;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    size_t bytes = 4096ul * 1024 * 1024;
    int *d; cudaMalloc(&d, bytes);

    int threads = 256;
    int n_warps = bytes / (32 * 1024);  // 32 KB per warp
    int blocks = n_warps / 8;  // 8 warps per block
    printf("# v8 per-warp coalesced: %d blocks × %d thr (%d warps total)\n",
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

    auto report = [&](float t, const char *name) {
        double bw = bytes/(t/1000)/1e9;
        printf("  %-30s %.3f ms  %.0f GB/s\n", name, t, bw);
    };

    printf("# %-30s %-12s %-12s\n", "method", "ms", "GB/s");
    report(bench([&]{ w_v8_coalesced<<<blocks, threads>>>(d); }), "v8 per-warp coalesced");
    report(bench([&]{ cudaMemsetAsync(d, 0xab, bytes, 0); }), "cudaMemset (ref)");

    return 0;
}
