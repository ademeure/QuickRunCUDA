// Custom memcpy via uint4 LDG+STG, compare to peak HBM BW
#ifndef MODE
#define MODE 0
#endif
#ifndef ELEMS
#define ELEMS 16
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    uint4* src = (uint4*)A;
    uint4* dst = (uint4*)C;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Single uint4 (16 B per thread per iter)
        int idx = (gid + i * total_threads) & 0xFFFFFF;
        dst[idx] = src[idx];
#elif MODE == 1
        // 4 unrolled uint4 per iter (64 B per thread per iter)
        int base = (gid * 4 + i * total_threads * 4) & 0xFFFFFFC;
        uint4 v0 = src[base];
        uint4 v1 = src[base + 1];
        uint4 v2 = src[base + 2];
        uint4 v3 = src[base + 3];
        dst[base] = v0;
        dst[base + 1] = v1;
        dst[base + 2] = v2;
        dst[base + 3] = v3;
#elif MODE == 2
        // 8 unrolled (128 B per thread per iter)
        int base = (gid * 8 + i * total_threads * 8) & 0xFFFFFF8;
        uint4 v0 = src[base], v1 = src[base+1], v2 = src[base+2], v3 = src[base+3];
        uint4 v4 = src[base+4], v5 = src[base+5], v6 = src[base+6], v7 = src[base+7];
        dst[base] = v0; dst[base+1] = v1; dst[base+2] = v2; dst[base+3] = v3;
        dst[base+4] = v4; dst[base+5] = v5; dst[base+6] = v6; dst[base+7] = v7;
#endif
    }
}
