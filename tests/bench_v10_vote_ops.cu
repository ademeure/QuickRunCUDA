// V10: warp vote/ballot primitives latency
#ifndef OP
#define OP 0  // 0=ballot, 1=vote.all, 2=vote.any, 3=vote.uni, 4=match.any
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    int lane = threadIdx.x;

    unsigned int v = lane + seed;
    unsigned long long t0, t1;
    if (lane == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    __syncwarp();

    #pragma unroll 16
    for (int i = 0; i < CHAIN_LEN; i++) {
#if OP == 0
        // ballot — returns 32-bit mask; need pred input
        unsigned int m;
        asm volatile("{.reg .pred P; setp.ne.u32 P, %1, 0; vote.sync.ballot.b32 %0, P, 0xFFFFFFFF;}"
                     : "=r"(m) : "r"(v & 1));
        v = m + 1;
#elif OP == 1
        unsigned int m;
        asm volatile("{.reg .pred P, Q; setp.ne.u32 P, %1, 0; vote.sync.all.pred Q, P, 0xFFFFFFFF; selp.u32 %0, 1, 0, Q;}"
                     : "=r"(m) : "r"(v));
        v += m;
#elif OP == 2
        unsigned int m;
        asm volatile("{.reg .pred P, Q; setp.ne.u32 P, %1, 0; vote.sync.any.pred Q, P, 0xFFFFFFFF; selp.u32 %0, 1, 0, Q;}"
                     : "=r"(m) : "r"(v));
        v += m;
#elif OP == 3
        unsigned int m;
        asm volatile("{.reg .pred P, Q; setp.ne.u32 P, %1, 0; vote.sync.uni.pred Q, P, 0xFFFFFFFF; selp.u32 %0, 1, 0, Q;}"
                     : "=r"(m) : "r"(v));
        v += m;
#elif OP == 4
        unsigned int m;
        asm volatile("match.any.sync.b32 %0, %1, 0xFFFFFFFF;"
                     : "=r"(m) : "r"(v));
        v ^= m;
#endif
    }

    __syncwarp();
    if (lane == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned*)C)[2] = v;
    }
}
