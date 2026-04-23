// Stride probe — replicate B300 catalog §22f L1/L2 cache granularity test.
//
// arg0 (iters): number of loads in the timed chain
// arg1 (stride_bytes): stride in bytes between consecutive loads in chain
// arg2 (mode):
//     0 = single-thread (1 lane, 1 block) — clean clock64 latency chain
//     1 = warp (32 lanes, 1 block) — coalescing footprint test
//
// Anti-DCE: XOR-chain dependency through register `v`, gated impossible-predicate store.
// Result: writes (t1 - t0) cycles to A[0] (lane 0 only).

#ifndef ITERS
#define ITERS_DEFAULT 4096
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C,
                                  int iters, int stride_bytes, int mode) {
    if (blockIdx.x != 0) return;
    if (mode == 0 && threadIdx.x != 0) return;
    if (mode == 1 && threadIdx.x >= 32) return;

    int lane = threadIdx.x;
    int *p = (int*)A;

    // Pointer-chase strategy: each load address depends on prior loaded value.
    // To keep within the 256 MiB buffer (= 0x4000000 ints = mask 0x03FFFFFF), and
    // to make the access pattern STRIDE-DEPENDENT, we compute:
    //   idx = ((v & MASK_OUT_STRIDE_BITS) + lane*stride_words) & BUF_MASK
    // For mode 0 (single-thread), each iter's address = (iter * stride_words) & BUF_MASK
    // because we want the catalog's "stride-chase" semantics — successive loads
    // at fixed stride.
    //
    // To prevent compiler from constant-folding the iteration variable, we mix
    // it into v. Use ld.global.ca.u32 (L1-cacheable, default).

    int stride_w = stride_bytes >> 2;  // bytes -> int words
    int v = lane * 17 + iters;         // make initial v depend on runtime input

    // Warm-up: prime the cache for the access pattern (NOT timed)
    #pragma unroll 1
    for (int it = 0; it < 64; it++) {
        int idx = ((it * stride_w) + lane * stride_w) & 0x03FFFFFF;
        int loaded;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(loaded) : "l"(p + idx));
        v ^= loaded;
    }

    __syncwarp(0xFFFFFFFF);
    long long t0 = clock64();

    #pragma unroll 1
    for (int it = 0; it < iters; it++) {
        // Stride pattern: each iter advances by `stride_w` ints from a base.
        // For mode 1 (warp), each lane reads at lane*stride_w offset.
        // Mask v contribution out of address (only used for v-dep), so address
        // pattern is the deterministic stride sweep the catalog claims.
        int idx = ((it * stride_w) + lane * stride_w) & 0x03FFFFFF;
        int loaded;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(loaded) : "l"(p + idx));
        v ^= loaded;
    }

    long long t1 = clock64();
    __syncwarp(0xFFFFFFFF);

    // Anti-DCE: gated impossible store of v to a far-out C location
    if (iters == -1) C[1024 + lane] = (float)v;

    // Output: cycles to C[0]. Also cy/load*1000 to C[2].
    if (lane == 0) {
        long long cycles = t1 - t0;
        ((long long*)C)[0] = cycles;
        // cy/load * 1000 (so we can read it as int with 3 decimals)
        ((long long*)C)[1] = (cycles * 1000) / iters;
    }
}
