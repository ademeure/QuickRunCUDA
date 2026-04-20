// IADD3 + predicate output cost.
// Mode 0: IADD3 no predicate
// Mode 1: IADD3 with carry-out predicate (use of predicate)
// Mode 2: IADD3 + extra ISETP after (separate inst)
// Mode 3: IMAD.IADD with predicate

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS], b[N_CHAINS], c[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (unsigned)(threadIdx.x * 131 + k * 17 + (unsigned)u2);
        b[k] = (unsigned)(threadIdx.x * 271 + k * 23);
        c[k] = (unsigned)(threadIdx.x * 419 + k * 41);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
#if MODE == 0
            // Plain IADD3: v = v + b + c
            asm volatile("add.u32 %0, %0, %1; add.u32 %0, %0, %2;"
                         : "+r"(v[k]) : "r"(b[k]), "r"(c[k]));
#elif MODE == 1
            // IADD3 with carry-out predicate (used in chain)
            asm volatile("{ .reg .pred p; "
                         "add.cc.u32 %0, %0, %1; "       // carry-set add
                         "addc.u32 %0, %0, %2; }"        // add-with-carry
                         : "+r"(v[k]) : "r"(b[k]), "r"(c[k]));
#elif MODE == 2
            // IADD3 + ISETP (predicate set as extra inst)
            asm volatile("{ .reg .pred p; "
                         "add.u32 %0, %0, %1; "
                         "add.u32 %0, %0, %2; "
                         "setp.ne.b32 p, %0, 0; "
                         "@p add.u32 %0, %0, 1; }"
                         : "+r"(v[k]) : "r"(b[k]), "r"(c[k]));
#elif MODE == 3
            // IMAD style: v = v * 1 + b (compiler may emit IMAD.IADD)
            asm volatile("mad.lo.u32 %0, %0, 1, %1;" : "+r"(v[k]) : "r"(b[k]));
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
