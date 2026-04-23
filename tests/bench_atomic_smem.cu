// Shared-memory atomic op cost characterization.
// Compare to global atomic (bench_atomic_types) — smem atomics use ATOMS/REDS,
// served at L1/SM scope, much faster than ATOMG/REDG (L2-served).

#ifndef OP
// 0=ADD 1=MIN 2=MAX 3=AND 4=OR 5=XOR 6=CAS 7=EXCH 8=SUB 9=INC
#define OP 0
#endif
#ifndef CONTENTION
// 0=uniq 1=broadcast 2=N=2
#define CONTENTION 0
#endif
#ifndef RETURN_VALUE
#define RETURN_VALUE 1
#endif
#ifndef ITERS
#define ITERS 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, float* B, float* C, int seed, int u1, int u2) {
    if (blockIdx.x != 0) return;
    int lane = threadIdx.x;

    __shared__ unsigned int smem[1024];
    if (lane == 0) {
        for (int i = 0; i < 1024; i++) smem[i] = 0;
    }
    __syncwarp();

    unsigned int* addr;
    #if CONTENTION == 0
        addr = &smem[lane * 32];   // 32 distinct lines (different banks)
    #elif CONTENTION == 1
        addr = &smem[0];
    #elif CONTENTION == 2
        addr = &smem[(lane & 1) * 32];
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
