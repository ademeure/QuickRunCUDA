// Atomic CAS spinlock patterns: contention, back-off variants.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef OP
#define OP 0
#endif
#ifndef N_LOCKERS
#define N_LOCKERS 148  // how many CTAs try to grab the lock
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (blockIdx.x >= N_LOCKERS) return;
    if (threadIdx.x != 0) return;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

#if OP == 0
    // Naive spinlock: loop until CAS succeeds
    while (atomicCAS(A, 0u, 1u) != 0u) {}
    // Hold briefly
    asm volatile("");
    atomicExch(A, 0u);  // release
#elif OP == 1
    // With pause-style backoff (__nanosleep)
    while (atomicCAS(A, 0u, 1u) != 0u) {
        __nanosleep(100);
    }
    atomicExch(A, 0u);
#elif OP == 2
    // Read-then-CAS (avoid CAS if obvious contention)
    while (true) {
        if (A[0] == 0) {
            if (atomicCAS(A, 0u, 1u) == 0u) break;
        }
    }
    atomicExch(A, 0u);
#elif OP == 3
    // Exponential backoff
    unsigned wait = 1;
    while (atomicCAS(A, 0u, 1u) != 0u) {
        for (unsigned j = 0; j < wait; j++) __nanosleep(10);
        if (wait < 256) wait *= 2;
    }
    atomicExch(A, 0u);
#endif

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    C[blockIdx.x] = (unsigned)(t1 - t0);
}
