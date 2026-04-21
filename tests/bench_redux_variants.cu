// redux.sync variants - which ops are supported and how fast?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned int v = (unsigned)threadIdx.x + (unsigned)u2;
    unsigned int total = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < ITERS; i++) {
        v += (unsigned)i;
        unsigned int r;
#if MODE == 0
        asm("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
#elif MODE == 1
        asm("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
#elif MODE == 2
        asm("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
#elif MODE == 3
        asm("redux.sync.and.b32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
#elif MODE == 4
        asm("redux.sync.or.b32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
#elif MODE == 5
        asm("redux.sync.xor.b32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
#elif MODE == 6
        asm("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
#endif
        total ^= r;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (total == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = total;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
