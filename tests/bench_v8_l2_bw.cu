// V8: L2 cache BW SoL
// Use buffer size 64 MB < L2 size (126 MB) → all accesses are L2 hits after warmup.
// Each thread reads float4 from buffer; stride ensures no L1 reuse within a thread.
// Theoretical L2 on B300 per prior catalog: ~23-36 TB/s range.

#ifndef K_INNER
#define K_INNER 64
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int THREADS = gridDim.x * blockDim.x;
    long long T = THREADS;

    // Buffer footprint 64 MB (fits in 126 MB L2 comfortably).
    // Coalesced access: warp reads 32 × 16 B = 512 B per cycle
    // L2 bypass L1 via ld.global.cg for pure L2 measurement
    unsigned int MASK = (4 * 1024 * 1024) - 1;  // 4M float4 = 64 MB

    float4 acc = make_float4(0, 0, 0, 0);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 8
        for (int k = 0; k < K_INNER; k++) {
            // Coalesced: consecutive threads hit consecutive float4
            // Each warp: 32 consecutive float4 = 512 B = 2 cache lines
            unsigned int idx = (gtid + (i * K_INNER + k) * THREADS) & MASK;
            float4 v;
            asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                         : "l"(A + idx));
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }

    if (acc.x == 1.234567e-30f) C[gtid] = acc;
}
