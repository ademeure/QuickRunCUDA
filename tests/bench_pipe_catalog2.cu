// Extended pipe catalog — min/max/sel/shuffle/video/etc.

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
    unsigned long long w[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        unsigned int lo = __float_as_int(1.0001f + 0.0001f*(threadIdx.x + k*23));
        unsigned int hi = __float_as_int(1.0002f + 0.0001f*(threadIdx.x + k*29));
        w[k] = ((unsigned long long)hi << 32) | lo;
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned int x = (unsigned int)w[k];
                float fv = __int_as_float(x);

#if OP == 20  // FMIN
                asm volatile("min.f32 %0, %0, %1;" : "+f"(fv) : "f"(1.5f));
#elif OP == 21  // FMAX
                asm volatile("max.f32 %0, %0, %1;" : "+f"(fv) : "f"(1.5f));
#elif OP == 22  // IMIN s32
                int ix = (int)x;
                asm volatile("min.s32 %0, %0, %1;" : "+r"(ix) : "r"(7));
                x = (unsigned int)ix;
#elif OP == 23  // IMAX s32
                int ix = (int)x;
                asm volatile("max.s32 %0, %0, %1;" : "+r"(ix) : "r"(7));
                x = (unsigned int)ix;
#elif OP == 24  // HMIN f16x2
                asm volatile("min.f16x2 %0, %0, %1;" : "+r"(x) : "r"(0x3C003C00u));
#elif OP == 25  // HMAX f16x2
                asm volatile("max.f16x2 %0, %0, %1;" : "+r"(x) : "r"(0x3C003C00u));
#elif OP == 26  // BFE (bit field extract)
                asm volatile("bfe.u32 %0, %0, 4, 16;" : "+r"(x));
#elif OP == 27  // BFI (bit field insert)
                asm volatile("bfi.b32 %0, %1, %0, 4, 8;" : "+r"(x) : "r"(0xAAu));
#elif OP == 28  // BREV
                asm volatile("brev.b32 %0, %0;" : "+r"(x));
#elif OP == 29  // CLZ
                unsigned int y;
                asm volatile("clz.b32 %0, %1;" : "=r"(y) : "r"(x));
                x = x + y;
#elif OP == 30  // SHF.L funnel shift
                asm volatile("shf.l.wrap.b32 %0, %0, %0, 7;" : "+r"(x));
#elif OP == 31  // SEL (predicated select)
                unsigned int y;
                asm volatile("{ .reg .pred p; setp.ne.u32 p, %0, 0; selp.u32 %0, 1, 2, p; }" : "+r"(x));
#elif OP == 32  // DP4A (int8 dot product)
                asm volatile("dp4a.s32.s32 %0, %0, %1, %0;" : "+r"(x) : "r"(0x01020304u));
#elif OP == 33  // DP2A (int16 dot product)
                asm volatile("dp2a.lo.s32.s32 %0, %0, %1, %0;" : "+r"(x) : "r"(0x01020304u));
#elif OP == 34  // MUFU.SIN
                fv = __int_as_float(x) * 0.01f + 0.5f;
                asm volatile("sin.approx.f32 %0, %0;" : "+f"(fv));
                x = __float_as_int(fv);
#elif OP == 35  // MUFU.COS
                fv = __int_as_float(x) * 0.01f + 0.5f;
                asm volatile("cos.approx.f32 %0, %0;" : "+f"(fv));
                x = __float_as_int(fv);
#elif OP == 36  // MUFU.LG2
                fv = __int_as_float(x) * 0.5f + 1.0f;
                asm volatile("lg2.approx.f32 %0, %0;" : "+f"(fv));
                x = __float_as_int(fv);
#elif OP == 37  // MUFU.TANH
                fv = __int_as_float(x) * 0.001f;
                asm volatile("tanh.approx.f32 %0, %0;" : "+f"(fv));
                x = __float_as_int(fv);
#elif OP == 38  // FSETP compare to predicate
                unsigned int y;
                asm volatile("{ .reg .pred p; setp.lt.f32 p, %1, 0f3F800000; selp.u32 %0, 1, 2, p; }" : "=r"(y) : "f"(fv));
                x = y;
#elif OP == 39  // IMUL.HI (wide high)
                asm volatile("mul.hi.u32 %0, %0, %1;" : "+r"(x) : "r"(0x5DEECE66Du));
#elif OP == 40  // VABSDIFF4 (video absolute diff)
                asm volatile("vabsdiff4.u32.u32.u32 %0, %0, %1, %0;" : "+r"(x) : "r"(0x12345678u));
#elif OP == 41  // SHFL sync (warp shuffle)
                asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1F, 0xFFFFFFFF;" : "+r"(x));
#elif OP == 42  // SAD (sum of absolute differences)
                asm volatile("sad.u32 %0, %0, %1, %0;" : "+r"(x) : "r"(0x12345678u));
#elif OP == 43  // NEG int (sub from 0)
                int ix = (int)x;
                asm volatile("neg.s32 %0, %0;" : "+r"(ix));
                x = (unsigned int)ix;
#elif OP == 44  // ABS int
                int ix = (int)x;
                asm volatile("abs.s32 %0, %0;" : "+r"(ix));
                x = (unsigned int)ix;
#elif OP == 45  // FABS
                asm volatile("abs.f32 %0, %0;" : "+f"(fv));
                x = __float_as_int(fv);
#elif OP == 46  // NOT
                asm volatile("not.b32 %0, %0;" : "+r"(x));
#endif
                w[k] = (w[k] & 0xFFFFFFFF00000000ULL) | x;
                (void)fv;
            }
        }
    }

    unsigned long long acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= w[k];
    if (acc == (unsigned long long)seed) ((unsigned long long*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
