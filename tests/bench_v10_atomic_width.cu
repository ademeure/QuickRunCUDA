// V10: 32-bit vs 64-bit atomic cost (common for counters/pointers)
#ifndef OP
#define OP 0  // 0=smem_u32, 1=smem_u64, 2=global_u32, 3=global_u64
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;
    __shared__ unsigned int s32[1];
    __shared__ unsigned long long s64[1];
    s32[0] = 0;
    s64[0] = 0;

    unsigned int v32 = 0;
    unsigned long long v64 = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
#if OP == 0
        v32 = atomicAdd(s32, v32 + 1);
#elif OP == 1
        v64 = atomicAdd(s64, v64 + 1);
#elif OP == 2
        v32 = atomicAdd((unsigned*)A, v32 + 1);
#elif OP == 3
        v64 = atomicAdd((unsigned long long*)A, v64 + 1);
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    ((unsigned long long*)C)[0] = t1 - t0;
    ((unsigned*)C)[2] = (unsigned)(v32 + v64);
}
