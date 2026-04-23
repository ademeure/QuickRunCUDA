// Audit: redux.sync latency (warp reduction)
// All 32 threads run; thread 0 measures via clock64
#ifndef CHAIN_LEN
#define CHAIN_LEN 4096
#endif
#ifndef OP
#define OP 0  // 0=add 1=min 2=max 3=or 4=and 5=xor
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C, int ITERS, int seed, int u2) {
    unsigned v = (unsigned)(threadIdx.x + seed + 1);

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
#if OP == 0
        asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v));
#elif OP == 1
        asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v));
#elif OP == 2
        asm volatile("redux.sync.max.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v));
#elif OP == 3
        asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v));
#elif OP == 4
        asm volatile("redux.sync.and.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v));
#elif OP == 5
        asm volatile("redux.sync.xor.b32 %0, %0, 0xFFFFFFFF;" : "+r"(v));
#endif
    }

    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned*)C)[2] = v;
    }
}
