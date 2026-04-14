// redux.sync latency chain (clock64) — varying mask widths + types.

#ifndef N_OPS
#define N_OPS 128
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 256
#endif
#ifndef OP
#define OP 0
#endif
#ifndef MASK
#define MASK 0xFFFFFFFF
#endif

extern "C" __global__ __launch_bounds__(32, 1) void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (blockIdx.x != 0) return;
    unsigned int u = threadIdx.x;
    int si = (int)threadIdx.x - 5;
    float fv = (float)threadIdx.x * 0.01f;
    unsigned long long total_dt = 0;

    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0 = 0, t1 = 0;
        if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        __syncwarp();
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0
            asm volatile("redux.sync.min.u32 %0, %0, %1;" : "+r"(u) : "n"(MASK));
#elif OP == 1
            asm volatile("redux.sync.max.u32 %0, %0, %1;" : "+r"(u) : "n"(MASK));
#elif OP == 2
            asm volatile("redux.sync.min.s32 %0, %0, %1;" : "+r"(si) : "n"(MASK));
#elif OP == 3
            asm volatile("redux.sync.min.f32 %0, %0, %1;" : "+f"(fv) : "n"(MASK));
#elif OP == 4
            asm volatile("redux.sync.min.NaN.f32 %0, %0, %1;" : "+f"(fv) : "n"(MASK));
#elif OP == 5
            asm volatile("redux.sync.add.u32 %0, %0, %1;" : "+r"(u) : "n"(MASK));
#elif OP == 6
            asm volatile("redux.sync.or.b32 %0, %0, %1;" : "+r"(u) : "n"(MASK));
#elif OP == 7
            asm volatile("redux.sync.and.b32 %0, %0, %1;" : "+r"(u) : "n"(MASK));
#elif OP == 8
            asm volatile("redux.sync.xor.b32 %0, %0, %1;" : "+r"(u) : "n"(MASK));
#endif
        }
        __syncwarp();
        if (threadIdx.x == 0) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
            total_dt += (t1 - t0);
        }
    }
    if (threadIdx.x == 0) {
        ((unsigned int*)C)[0] = u + (unsigned)si + __float_as_int(fv);
        ((unsigned long long*)C)[1] = total_dt;
    }
}
