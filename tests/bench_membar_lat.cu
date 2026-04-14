// Memory fence latency — clock64-bracketed.

#ifndef N_OPS
#define N_OPS 64
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 256
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned long long total_dt = 0;

    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0
            asm volatile("membar.cta;");
#elif OP == 1
            asm volatile("membar.gl;");
#elif OP == 2
            asm volatile("membar.sys;");
#elif OP == 3
            asm volatile("fence.acq_rel.cta;");
#elif OP == 4
            asm volatile("fence.sc.cta;");
#elif OP == 5
            asm volatile("fence.acq_rel.gpu;");
#elif OP == 6
            asm volatile("fence.sc.gpu;");
#elif OP == 7
            asm volatile("fence.acquire.cluster;");
#elif OP == 8
            asm volatile("bar.warp.sync -1;");
#elif OP == 9
            asm volatile("bar.sync 0;");
#elif OP == 10
            asm volatile("nanosleep.u32 10;");
#elif OP == 11
            asm volatile("nanosleep.u32 100;");
#endif
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }
    ((unsigned long long*)C)[1] = total_dt;
}
