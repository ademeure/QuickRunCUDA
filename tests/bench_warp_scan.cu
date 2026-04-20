// 32-lane warp scan (prefix sum) speed-of-light
#ifndef MODE
#define MODE 0
#endif
#ifndef N_ITERS
#define N_ITERS 100000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)threadIdx.x + 1u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_ITERS; i++) {
        v += (unsigned)i + (unsigned)u2;
#if MODE == 0
        // Hillis-Steele warp scan via __shfl_up_sync (5 steps)
        unsigned int t;
        t = __shfl_up_sync(0xFFFFFFFF, v,  1); if (threadIdx.x >=  1) v += t;
        t = __shfl_up_sync(0xFFFFFFFF, v,  2); if (threadIdx.x >=  2) v += t;
        t = __shfl_up_sync(0xFFFFFFFF, v,  4); if (threadIdx.x >=  4) v += t;
        t = __shfl_up_sync(0xFFFFFFFF, v,  8); if (threadIdx.x >=  8) v += t;
        t = __shfl_up_sync(0xFFFFFFFF, v, 16); if (threadIdx.x >= 16) v += t;
#elif MODE == 1
        // PTX shfl.up with mask (no branch)
        unsigned int t;
        asm("shfl.sync.up.b32 %0, %1,  1, 0, 0xffffffff;" : "=r"(t) : "r"(v)); if (threadIdx.x >=  1) v += t;
        asm("shfl.sync.up.b32 %0, %1,  2, 0, 0xffffffff;" : "=r"(t) : "r"(v)); if (threadIdx.x >=  2) v += t;
        asm("shfl.sync.up.b32 %0, %1,  4, 0, 0xffffffff;" : "=r"(t) : "r"(v)); if (threadIdx.x >=  4) v += t;
        asm("shfl.sync.up.b32 %0, %1,  8, 0, 0xffffffff;" : "=r"(t) : "r"(v)); if (threadIdx.x >=  8) v += t;
        asm("shfl.sync.up.b32 %0, %1, 16, 0, 0xffffffff;" : "=r"(t) : "r"(v)); if (threadIdx.x >= 16) v += t;
#elif MODE == 2
        // Predicated PTX shfl.up with built-in lane filter
        unsigned int t;
        asm("{ .reg .pred p; shfl.sync.up.b32 %0|p, %1,  1, 0, 0xffffffff; @p add.u32 %1, %1, %0; }" : "=r"(t), "+r"(v) :);
        asm("{ .reg .pred p; shfl.sync.up.b32 %0|p, %1,  2, 0, 0xffffffff; @p add.u32 %1, %1, %0; }" : "=r"(t), "+r"(v) :);
        asm("{ .reg .pred p; shfl.sync.up.b32 %0|p, %1,  4, 0, 0xffffffff; @p add.u32 %1, %1, %0; }" : "=r"(t), "+r"(v) :);
        asm("{ .reg .pred p; shfl.sync.up.b32 %0|p, %1,  8, 0, 0xffffffff; @p add.u32 %1, %1, %0; }" : "=r"(t), "+r"(v) :);
        asm("{ .reg .pred p; shfl.sync.up.b32 %0|p, %1, 16, 0, 0xffffffff; @p add.u32 %1, %1, %0; }" : "=r"(t), "+r"(v) :);
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_ITERS=%d clk=%llu cy/scan=%.3f\n",
               MODE, N_ITERS, t1 - t0, (double)(t1-t0)/(double)N_ITERS);
    }
}
