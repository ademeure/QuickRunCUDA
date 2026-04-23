// Atomic op type sweep: cost vs op type × return value × coalescing.
// 1 CTA, 32 threads, ITERS atomic ops per thread (configurable contention).

#ifndef OP
// 0 = atomicAdd  (ADD)
// 1 = atomicMin  (MIN)
// 2 = atomicMax  (MAX)
// 3 = atomicAnd  (AND)
// 4 = atomicOr   (OR)
// 5 = atomicXor  (XOR)
// 6 = atomicCAS  (CAS)
// 7 = atomicExch (EXCH)
// 8 = atomicSub  (SUB)
// 9 = atomicInc  (INC saturating)
#define OP 0
#endif

#ifndef CONTENTION
// 0 = no contention (each thread has its own address)
// 1 = warp-broadcast (all 32 lanes hit same address — coalesced)
// 2 = N=2 split (lane%2 picks 1 of 2)
#define CONTENTION 0
#endif

#ifndef RETURN_VALUE
#define RETURN_VALUE 1   // 1 = use return (ATOMG), 0 = no return (REDG)
#endif

#ifndef ITERS
#define ITERS 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, float* B, float* C, int seed, int u1, int u2) {
    if (blockIdx.x != 0) return;
    int lane = threadIdx.x;
    unsigned int* base = A;

    unsigned int* addr;
    #if CONTENTION == 0
        addr = base + lane * 32;  // 32 distinct lines, each thread unique addr
    #elif CONTENTION == 1
        addr = base;              // all 32 lanes broadcast to same addr
    #elif CONTENTION == 2
        addr = base + (lane & 1) * 32;  // 2 distinct addrs, 16 lanes each
    #endif

    unsigned int acc = lane;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if RETURN_VALUE == 1
    #if OP == 0
        acc ^= atomicAdd(addr, 1u);
    #elif OP == 1
        acc ^= atomicMin(addr, acc);
    #elif OP == 2
        acc ^= atomicMax(addr, acc);
    #elif OP == 3
        acc ^= atomicAnd(addr, ~(unsigned)i);
    #elif OP == 4
        acc ^= atomicOr(addr, (unsigned)i);
    #elif OP == 5
        acc ^= atomicXor(addr, (unsigned)i);
    #elif OP == 6
        acc ^= atomicCAS(addr, acc, acc + 1u);
    #elif OP == 7
        acc ^= atomicExch(addr, (unsigned)i);
    #elif OP == 8
        acc ^= atomicSub(addr, 1u);
    #elif OP == 9
        acc ^= atomicInc(addr, 0xFFFFFFFFu);
    #endif
#else
    // No-return variants compile to REDG when supported
    #if OP == 0
        atomicAdd(addr, 1u);
    #elif OP == 1
        atomicMin(addr, acc);
    #elif OP == 2
        atomicMax(addr, acc);
    #elif OP == 3
        atomicAnd(addr, ~(unsigned)i);
    #elif OP == 4
        atomicOr(addr, (unsigned)i);
    #elif OP == 5
        atomicXor(addr, (unsigned)i);
    #elif OP == 6
        atomicCAS(addr, acc, acc + 1u);
    #elif OP == 7
        atomicExch(addr, (unsigned)i);
    #elif OP == 8
        atomicSub(addr, 1u);
    #elif OP == 9
        atomicInc(addr, 0xFFFFFFFFu);
    #endif
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (acc == 0xDEADBEEF) C[lane] = (float)acc;
    if (lane == 0) {
        ((unsigned long long*)C)[1024] = (unsigned long long)(t1 - t0);
    }
}
