// Branch / predication cost: divergent vs uniform vs predicated.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 1024
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned acc = seed;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0
        // No branch — straight-line
        acc = acc * 1664525u + 1013904223u + i;
#elif OP == 1
        // Predicated select (no branch)
        unsigned new_acc = acc * 1664525u + 1013904223u + i;
        acc = (i & 1) ? new_acc : acc;
#elif OP == 2
        // UNIFORM branch (all threads take same path)
        if (i & 1) {
            acc = acc * 1664525u + 1013904223u + i;
        } else {
            acc = acc + i;
        }
#elif OP == 3
        // DIVERGENT branch — half lanes take different path
        if (tid & 1) {
            acc = acc * 1664525u + 1013904223u + i;
        } else {
            acc = acc + i;
        }
#elif OP == 4
        // Heavily divergent — each lane different path
        switch (tid & 7) {
            case 0: acc = acc + i; break;
            case 1: acc = acc * i; break;
            case 2: acc = acc - i; break;
            case 3: acc = acc ^ i; break;
            case 4: acc = acc | i; break;
            case 5: acc = acc & i; break;
            case 6: acc = acc << 1; break;
            case 7: acc = acc >> 1; break;
        }
#elif OP == 5
        // Loop with constant trip-count (compiler unrolls fully)
        for (int j = 0; j < 4; j++) acc = acc * 1664525u + j;
#elif OP == 6
        // Loop with runtime trip-count (compiler can't unroll)
        for (int j = 0; j < (i & 3); j++) acc = acc * 1664525u + j;
#endif
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
    }
}
