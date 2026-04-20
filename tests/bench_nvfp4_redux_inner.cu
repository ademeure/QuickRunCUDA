// Isolate the warp-reduce step: do it INSIDE the timing loop.
// This shows the redux.sync.add advantage realistically when the
// reduction itself is the bottleneck (e.g., per-tile streaming reduce).

#ifndef MODE
#define MODE 0
#endif
#ifndef N_ITERS
#define N_ITERS 10000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Each thread holds a small per-thread accumulator that gets warp-reduced
    // many times in a tight loop.
    float fv = (float)threadIdx.x + 0.5f;
    int   iv = threadIdx.x + 1;
    float ftotal = 0.0f;
    int   itotal = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        // Add some per-iter perturbation (defeats compiler hoist)
        fv += (float)(it * (1 + (unsigned)u2));
        iv += it * (1 + (unsigned)u2);

#if MODE == 0
        // FP32 SHFL chain reduction
        float r = fv;
        r += __shfl_xor_sync(0xFFFFFFFF, r, 16);
        r += __shfl_xor_sync(0xFFFFFFFF, r,  8);
        r += __shfl_xor_sync(0xFFFFFFFF, r,  4);
        r += __shfl_xor_sync(0xFFFFFFFF, r,  2);
        r += __shfl_xor_sync(0xFFFFFFFF, r,  1);
        ftotal += r;
#elif MODE == 1
        // Int redux.sync.add
        int r;
        asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(iv));
        itotal += r;
#elif MODE == 2
        // FP32 + 5-step PTX SHFL bfly (manual)
        float r = fv;
        float y;
        asm volatile("shfl.sync.bfly.b32 %0, %1, 16, 0x1f, 0xffffffff;" : "=f"(y) : "f"(r)); r += y;
        asm volatile("shfl.sync.bfly.b32 %0, %1,  8, 0x1f, 0xffffffff;" : "=f"(y) : "f"(r)); r += y;
        asm volatile("shfl.sync.bfly.b32 %0, %1,  4, 0x1f, 0xffffffff;" : "=f"(y) : "f"(r)); r += y;
        asm volatile("shfl.sync.bfly.b32 %0, %1,  2, 0x1f, 0xffffffff;" : "=f"(y) : "f"(r)); r += y;
        asm volatile("shfl.sync.bfly.b32 %0, %1,  1, 0x1f, 0xffffffff;" : "=f"(y) : "f"(r)); r += y;
        ftotal += r;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)ftotal == seed && itotal == seed) ((unsigned*)C)[blockIdx.x] = itotal;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_ITERS=%d clk=%llu cy/iter=%.3f\n",
               MODE, N_ITERS, t1 - t0, (double)(t1-t0)/(double)N_ITERS);
    }
}
