// V9: Atomic latency — global memory atomicAdd dependency chain
// Single thread, each atomic depends on previous result
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif
#ifndef SCOPE
#define SCOPE 0  // 0=global default, 1=cta (block), 2=gpu (multi-GPU-aware)
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    // Each iter: v = atomicAdd(&A[0], 1) - chains via return value
    unsigned int v = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 16
    for (int i = 0; i < CHAIN_LEN; i++) {
#if SCOPE == 0
        // Default scope (system)
        v = atomicAdd(&A[v & 0xFFFF], 1);
#elif SCOPE == 1
        // CTA scope
        asm volatile("atom.cta.add.u32 %0, [%1], 1;" : "=r"(v) : "l"(A));
#elif SCOPE == 2
        // GPU scope
        asm volatile("atom.gpu.add.u32 %0, [%1], 1;" : "=r"(v) : "l"(A));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    ((unsigned long long*)C)[0] = t1 - t0;
    ((unsigned int*)C)[2] = v;
}
