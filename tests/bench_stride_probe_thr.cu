// Throughput-oriented stride sweep — replicate the catalog's "56 cy/load at small stride" claim.
//
// arg0 (iters): outer iterations (number of unrolled blocks of K_INNER independent loads)
// arg1 (stride_bytes): stride between consecutive loads
// arg2 (mode): 0 = single-thread, 1 = full warp
//
// Strategy: K_INNER independent loads per iter, each into its own register, then sum into accumulator.
// This breaks the dep chain so HW can pipeline LDGs and approach throughput limits.

#ifndef K_INNER
#define K_INNER 16
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C,
                                  int iters, int stride_bytes, int mode) {
    if (blockIdx.x != 0) return;
    if (mode == 0 && threadIdx.x != 0) return;
    if (mode == 1 && threadIdx.x >= 32) return;

    int lane = threadIdx.x;
    int *p = (int*)A;
    int stride_w = stride_bytes >> 2;
    int v_init = lane * 17 + iters;
    int acc = v_init;

    // Warm-up
    #pragma unroll 1
    for (int it = 0; it < 64; it++) {
        int idx = ((it * K_INNER * stride_w) + lane * stride_w) & 0x03FFFFFF;
        int loaded;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(loaded) : "l"(p + idx));
        acc ^= loaded;
    }

    __syncwarp(0xFFFFFFFF);
    long long t0 = clock64();

    #pragma unroll 1
    for (int it = 0; it < iters; it++) {
        long long base = (long long)it * K_INNER * stride_w + (long long)lane * stride_w;
        // K_INNER independent loads
        int r[K_INNER];
        #pragma unroll
        for (int k = 0; k < K_INNER; k++) {
            int idx = (int)((base + (long long)k * stride_w) & 0x03FFFFFF);
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(r[k]) : "l"(p + idx));
        }
        #pragma unroll
        for (int k = 0; k < K_INNER; k++) {
            acc ^= r[k];
        }
    }

    long long t1 = clock64();
    __syncwarp(0xFFFFFFFF);

    if (iters == -1) C[1024 + lane] = (float)acc;

    if (lane == 0) {
        long long cycles = t1 - t0;
        long long total_loads = (long long)iters * K_INNER;
        ((long long*)C)[0] = cycles;
        ((long long*)C)[1] = (cycles * 1000) / total_loads;  // cy/load * 1000
    }
}
