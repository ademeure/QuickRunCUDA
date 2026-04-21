// V10: atomicAdd / atomicMin / atomicMax / atomicExch comparison
// Single thread, serial chain
#ifndef OP
#define OP 0  // 0=add, 1=min, 2=max, 3=exch, 4=or, 5=xor
#endif
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
    __shared__ unsigned int smem[1];
    smem[0] = 0;

#if SCOPE == 0
    unsigned int* loc = smem;
#else
    unsigned int* loc = A;
#endif

    unsigned int v = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
#if OP == 0
        v = atomicAdd(loc, v + 1);
#elif OP == 1
        v = atomicMin(loc, v + 1);
#elif OP == 2
        v = atomicMax(loc, v + 1);
#elif OP == 3
        v = atomicExch(loc, v + 1);
#elif OP == 4
        v = atomicOr(loc, v + 1);
#elif OP == 5
        v = atomicXor(loc, v + 1);
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    ((unsigned long long*)C)[0] = t1 - t0;
    ((unsigned*)C)[2] = v;
}
