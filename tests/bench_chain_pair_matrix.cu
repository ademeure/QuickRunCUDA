// Pair-wise chain latency matrix.
// OP_A and OP_B: 0=FFMA, 1=IMAD, 2=LOP3, 3=IADD3, 4=SHF, 5=PRMT
// Tests pair (OP_A then OP_B) chained through register v.

#ifndef N_INNER
#define N_INNER 1000
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif
#ifndef OP_A
#define OP_A 0
#endif
#ifndef OP_B
#define OP_B 0
#endif

#define INST_FFMA(d) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(d) : "f"(fb), "f"(fc))
#define INST_IMAD(d) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(d) : "r"(b), "r"(c))
#define INST_LOP3(d) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(d) : "r"(b), "r"(c))
#define INST_IADD3(d) asm volatile("add.u32 %0, %0, %1; add.u32 %0, %0, %2;" : "+r"(d) : "r"(b), "r"(c))
#define INST_SHF(d) asm volatile("shf.l.wrap.b32 %0, %0, %1, 5;" : "+r"(d) : "r"(b))
#define INST_PRMT(d) asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(d) : "r"(b), "r"(c))

#define DISPATCH(op, d, fd) do { \
    if (op == 0) { fd = __uint_as_float(d); INST_FFMA(fd); d = __float_as_uint(fd); } \
    else if (op == 1) INST_IMAD(d); \
    else if (op == 2) INST_LOP3(d); \
    else if (op == 3) INST_IADD3(d); \
    else if (op == 4) INST_SHF(d); \
    else if (op == 5) INST_PRMT(d); \
} while (0)

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)threadIdx.x + 1u;
    // Runtime-unknown b, c via u2 to defeat compiler folding
    unsigned int b = 0xDEADBEEFu ^ (unsigned)u2;
    unsigned int c = 0xC0FFEE00u ^ ((unsigned)u2 << 8);
    float fv, fb = 1.0000001f + (float)u2 * 1e-9f, fc = 0.0f + (float)u2 * 1e-9f;
    (void)fv;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            DISPATCH(OP_A, v, fv);
            DISPATCH(OP_B, v, fv);
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("OP_A=%d OP_B=%d pairs=%llu clk=%llu cy/pair=%.4f\n",
               OP_A, OP_B, total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
