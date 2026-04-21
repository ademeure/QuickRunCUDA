// V6 J4: Shuffle variant latency
// MODE 0: shfl.sync.bfly (XOR)
// MODE 1: shfl.sync.up
// MODE 2: shfl.sync.down
// MODE 3: shfl.sync.idx
// 8 chained shuffles per iter
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = threadIdx.x + 1;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
#if MODE == 0
            asm volatile("{ .reg .pred p; shfl.sync.bfly.b32 %0|p, %0, 1, 0x1f, 0xffffffff; }" : "+r"(v));
#elif MODE == 1
            asm volatile("{ .reg .pred p; shfl.sync.up.b32 %0|p, %0, 1, 0x0, 0xffffffff; }" : "+r"(v));
#elif MODE == 2
            asm volatile("{ .reg .pred p; shfl.sync.down.b32 %0|p, %0, 1, 0x1f, 0xffffffff; }" : "+r"(v));
#elif MODE == 3
            // XOR-back to keep v thread-dependent
            unsigned int tmp;
            asm volatile("{ .reg .pred p; shfl.sync.idx.b32 %0|p, %1, %2, 0x1f, 0xffffffff; }" : "=r"(tmp) : "r"(v), "r"((unsigned)k));
            v = v ^ tmp;
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == 0xCAFEBABE) C[blockIdx.x] = (float)v;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f cy/shfl=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/8.0);
    }
}
