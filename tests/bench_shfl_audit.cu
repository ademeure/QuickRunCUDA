// SHFL latency + throughput audit. Identical surrounding code for both forms.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 4096
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    unsigned v = tid + seed;
    unsigned acc = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0
        // IDX with constant (no IMAD needed for index)
        v = __shfl_sync(0xFFFFFFFF, v, 1);
#elif OP == 1
        v = __shfl_xor_sync(0xFFFFFFFF, v, 1);
#elif OP == 2
        // IDX with computed index (forces IMAD)
        v = __shfl_sync(0xFFFFFFFF, v, (i+1) & 31);
#elif OP == 3
        // 8 independent SHFLs in parallel — throughput test
        unsigned v0 = __shfl_xor_sync(0xFFFFFFFF, v, 1);
        unsigned v1 = __shfl_xor_sync(0xFFFFFFFF, v, 2);
        unsigned v2 = __shfl_xor_sync(0xFFFFFFFF, v, 4);
        unsigned v3 = __shfl_xor_sync(0xFFFFFFFF, v, 8);
        unsigned v4 = __shfl_xor_sync(0xFFFFFFFF, v, 16);
        unsigned v5 = __shfl_xor_sync(0xFFFFFFFF, v, 3);
        unsigned v6 = __shfl_xor_sync(0xFFFFFFFF, v, 5);
        unsigned v7 = __shfl_xor_sync(0xFFFFFFFF, v, 6);
        v = v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7;
#elif OP == 4
        // 8 independent IDX — throughput
        unsigned v0 = __shfl_sync(0xFFFFFFFF, v, 1);
        unsigned v1 = __shfl_sync(0xFFFFFFFF, v, 2);
        unsigned v2 = __shfl_sync(0xFFFFFFFF, v, 3);
        unsigned v3 = __shfl_sync(0xFFFFFFFF, v, 4);
        unsigned v4 = __shfl_sync(0xFFFFFFFF, v, 5);
        unsigned v5 = __shfl_sync(0xFFFFFFFF, v, 6);
        unsigned v6 = __shfl_sync(0xFFFFFFFF, v, 7);
        unsigned v7 = __shfl_sync(0xFFFFFFFF, v, 8);
        v = v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7;
#endif
        acc ^= v;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
        C[3] = v;
    }
}
