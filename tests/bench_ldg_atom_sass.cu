// Compare __ldg vs ld.global.ca SASS encoding, and red vs atom SASS.
// Just emit both and look at the dump.

#ifndef OP
#define OP 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* ai = (unsigned int*)A;
    unsigned int* ci = (unsigned int*)C;
    unsigned int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FFF;

    unsigned int v = 0;
#if OP == 0
    // __ldg intrinsic
    v = __ldg(ai + idx);
#elif OP == 1
    // PTX ld.global.ca
    asm("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(ai + idx));
#elif OP == 2
    // PTX ld.global.cg (cache global, bypass L1)
    asm("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(ai + idx));
#elif OP == 3
    // PTX ld.global (default - matches ld.global.ca usually)
    asm("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(ai + idx));
#elif OP == 4
    // atomicAdd, return value used
    v = atomicAdd(ai + idx, (unsigned)seed);
#elif OP == 5
    // PTX atom.global.add (with return)
    asm("atom.global.add.u32 %0, [%1], %2;" : "=r"(v) : "l"(ai + idx), "r"(seed));
#elif OP == 6
    // PTX red.global.add (no return)
    asm("red.global.add.u32 [%0], %1;" :: "l"(ai + idx), "r"(seed));
    v = 0;
#elif OP == 7
    // PTX red.relaxed.gpu.global.add
    asm("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(ai + idx), "r"(seed));
    v = 0;
#endif

    ci[idx] = v;
}
