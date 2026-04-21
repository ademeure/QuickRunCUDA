// V9: Pipelined atomic throughput — multiple in-flight atomics
// N_CHAINS independent atomics per iter, no dep chain between them
#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    // N_CHAINS independent atomic chains (each always hits separate addr)
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) v[k] = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 16
    for (int i = 0; i < CHAIN_LEN; i++) {
        // Independent atomics (no chain between them — HW can pipeline)
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
            v[k] += atomicAdd(&A[k], 1);
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    ((unsigned long long*)C)[0] = t1 - t0;
    unsigned int sum = 0;
    for (int k = 0; k < N_CHAINS; k++) sum ^= v[k];
    ((unsigned int*)C)[2] = sum;  // Anti-DCE
}
