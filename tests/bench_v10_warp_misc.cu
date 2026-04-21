// V10: __activemask, __popc, __ffs, __brev, __clz cost
#ifndef OP
#define OP 0  // 0=activemask, 1=popc, 2=ffs, 3=brev, 4=clz, 5=mov
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)threadIdx.x + (unsigned)seed;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    __syncwarp();

    #pragma unroll 16
    for (int i = 0; i < CHAIN_LEN; i++) {
#if OP == 0
        // activemask — get current mask
        unsigned int m;
        asm volatile("activemask.b32 %0;" : "=r"(m));
        v += m;
#elif OP == 1
        // popc — population count
        v = __popc(v);
        v += i;
#elif OP == 2
        // ffs — find first set bit
        v = __ffs(v);
        v += i;
#elif OP == 3
        // brev — bit reverse
        v = __brev(v);
        v += i;
#elif OP == 4
        // clz — count leading zeros
        v = __clz(v);
        v += i;
#elif OP == 5
        // baseline: just MOV
        asm volatile("mov.u32 %0, %1;" : "=r"(v) : "r"(v + 1));
#endif
    }

    __syncwarp();
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned*)C)[2] = v;
    }
}
