// V10: atomicCAS latency — uncontended vs contended (for spinlock patterns)
// Single thread doing successful CAS: no-op in steady state (compare always succeeds)
#ifndef SCOPE
#define SCOPE 0  // 0=smem, 1=global
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    __shared__ unsigned int slock;
    slock = 0;

    unsigned int expected = 0;
    unsigned int old = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
#if SCOPE == 0
        // SMEM CAS
        old = atomicCAS(&slock, expected, old + 1);
#elif SCOPE == 1
        // Global CAS
        old = atomicCAS(&A[0], expected, old + 1);
#endif
        expected = old + 1;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    ((unsigned long long*)C)[0] = t1 - t0;
    ((unsigned*)C)[2] = old;
}
