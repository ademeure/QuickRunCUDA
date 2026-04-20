// Integer/bit-op pipe throughput sweep at 1500 MHz.
// Modes:
//   0: BREV (bit reverse)
//   1: POPC (population count)
//   2: BMSK (bit mask gen — generated from PTX bfe?)
//   3: SHF.L (funnel left shift)
//   4: SHF.R (funnel right shift)
//   5: PRMT (byte permute, mode 0)
//   6: FLO (find leading one == clz then bit-flip)
//   7: SHFL (warp shuffle)
//   8: BFI (bit field insert via PTX prmt actually;
//          PTX has bfi but inline asm needed)
//   9: SAD (sum-absolute-difference)

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP_MODE
#define OP_MODE 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS];
    unsigned int b_src[N_CHAINS];
    unsigned int c_src[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k]     = 0xDEAD0000u + (threadIdx.x * 131 + k * 17);
        b_src[k] = 0xBEEF0000u + (threadIdx.x * 271 + k * 23);
        c_src[k] = 0xCAFE0000u + (threadIdx.x * 419 + k * 41);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if OP_MODE == 0
                asm volatile("brev.b32 %0, %0;" : "+r"(v[k]));
#elif OP_MODE == 1
                asm volatile("popc.b32 %0, %0;" : "+r"(v[k]));
#elif OP_MODE == 2
                // bit field extract — PTX bfe.u32
                asm volatile("bfe.u32 %0, %0, 4, 16;" : "+r"(v[k]));
#elif OP_MODE == 3
                // funnel shift left
                asm volatile("shf.l.wrap.b32 %0, %0, %1, 5;" : "+r"(v[k]) : "r"(b_src[k]));
#elif OP_MODE == 4
                // funnel shift right
                asm volatile("shf.r.wrap.b32 %0, %0, %1, 5;" : "+r"(v[k]) : "r"(b_src[k]));
#elif OP_MODE == 5
                // byte permute
                asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(v[k]) : "r"(b_src[k]), "r"(c_src[k]));
#elif OP_MODE == 6
                // count leading zeros (FLO)
                asm volatile("clz.b32 %0, %0;" : "+r"(v[k]));
#elif OP_MODE == 7
                // SHFL — read from varying lane (encoded in b_src), real chain dep
                asm volatile("shfl.sync.idx.b32 %0, %0, %1, 0x1f, 0xffffffff;" : "+r"(v[k]) : "r"(b_src[k] & 0x1f));
#elif OP_MODE == 8
                // bit field insert
                asm volatile("bfi.b32 %0, %0, %1, 4, 16;" : "+r"(v[k]) : "r"(b_src[k]));
#elif OP_MODE == 9
                // PTX vabsdiff2 EMULATED on B300 — compiles to ~9 SASS inst
                // (PRMT + SHF + IMAD.IADD + IABS chain), don't trust as native pipe
                asm volatile("vabsdiff2.s32.s32.s32 %0, %0, %1, %2;" : "+r"(v[k]) : "r"(b_src[k]), "r"(c_src[k]));
#endif
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if ((int)acc == seed)
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
