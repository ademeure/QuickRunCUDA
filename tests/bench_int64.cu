// 64-bit integer op catalog.

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned long long v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++)
        v[k] = 0xDEADBEEF12345678ULL ^ (threadIdx.x * 0x100000007ULL + k * 17);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned long long x = v[k];
                unsigned long long n = v[(k+1) & (N_CHAINS-1)];
#if OP == 0   // u64 ADD
                asm volatile("add.u64 %0, %0, %1;" : "+l"(x) : "l"(n));
#elif OP == 1 // u64 SUB
                asm volatile("sub.u64 %0, %0, %1;" : "+l"(x) : "l"(n));
#elif OP == 2 // u64 MUL.LO
                asm volatile("mul.lo.u64 %0, %0, %1;" : "+l"(x) : "l"(n));
#elif OP == 3 // u64 MUL.HI
                asm volatile("mul.hi.u64 %0, %0, %1;" : "+l"(x) : "l"(n));
#elif OP == 4 // u64 MAD.LO
                asm volatile("mad.lo.u64 %0, %0, %1, %1;" : "+l"(x) : "l"(n));
#elif OP == 5 // u64 AND
                asm volatile("and.b64 %0, %0, %1;" : "+l"(x) : "l"(n));
#elif OP == 6 // u64 XOR
                asm volatile("xor.b64 %0, %0, %1;" : "+l"(x) : "l"(n));
#elif OP == 7 // u64 SHL
                asm volatile("shl.b64 %0, %0, %1;" : "+l"(x) : "r"((unsigned)(n & 0x3F)));
#elif OP == 8 // u64 SHR
                asm volatile("shr.u64 %0, %0, %1;" : "+l"(x) : "r"((unsigned)(n & 0x3F)));
#elif OP == 9 // u64 MIN
                asm volatile("min.u64 %0, %0, %1;" : "+l"(x) : "l"(n));
#elif OP == 10 // IADD.CC + IADD.X (explicit 2-inst u64 add)
                unsigned int lo = (unsigned int)x, hi = (unsigned int)(x>>32);
                unsigned int nlo = (unsigned int)n, nhi = (unsigned int)(n>>32);
                asm volatile("{ add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3; }"
                             : "+r"(lo), "+r"(hi) : "r"(nlo), "r"(nhi));
                x = (unsigned long long)lo | ((unsigned long long)hi << 32);
#endif
                v[k] = x;
            }
        }
    }
    unsigned long long acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if (acc == (unsigned long long)seed) ((unsigned long long*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
