// Shared broadcast, data-dependent chain (no DCE).

#ifndef N_OPS
#define N_OPS 128
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 256
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Full warp participates for broadcast fast-path test
    if (blockIdx.x != 0) return;

    // Initialize: smem[0] = small value so pointer-chase stays bounded
    if (threadIdx.x == 0) {
        for (int i = 0; i < 128; i++) smem[i] = (i + 1) & 0x7F;
    }
    __syncwarp();

    unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned int idx = 0;
    unsigned long long total_dt = 0;

    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0 = 0, t1 = 0;
        if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        __syncwarp();

        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0  // BROADCAST: all lanes read smem[0] (which may be data-dep)
            unsigned int r;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(base));
            idx = r;
#elif OP == 1  // Per-lane dep-chase
            unsigned int r;
            unsigned int addr = base + ((idx & 0x7F) << 2);
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(addr));
            idx = r;
#elif OP == 2  // All lanes chase through same chain (broadcast + dep)
            unsigned int r;
            unsigned int addr = base + ((idx & 0x7F) << 2);
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(addr));
            idx = __shfl_sync(0xFFFFFFFF, r, 0);  // broadcast lane-0 result
#endif
        }

        __syncwarp();
        if (threadIdx.x == 0) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
            total_dt += (t1 - t0);
        }
    }
    if (threadIdx.x == 0) {
        ((unsigned int*)C)[0] = idx;
        ((unsigned long long*)C)[1] = total_dt;
    }
}
