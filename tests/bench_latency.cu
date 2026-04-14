// Latency probes: single-chain back-to-back dependency
// measures min cycles between issue and result-availability.
// Use 1 thread, 1 warp, 1 block to get pure latency with minimal ILP.

#ifndef UNROLL
#define UNROLL 64
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int x = (unsigned int)seed;
    float f = __int_as_float(x);
    unsigned long long lx = (unsigned long long)seed;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // FFMA latency
            asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(f));
#elif OP == 1  // FMUL
            asm volatile("mul.rn.f32 %0, %0, %0;" : "+f"(f));
#elif OP == 2  // FADD
            asm volatile("add.rn.f32 %0, %0, %0;" : "+f"(f));
#elif OP == 3  // IMAD
            asm volatile("mad.lo.u32 %0, %0, %0, %0;" : "+r"(x));
#elif OP == 4  // IADD3
            asm volatile("add.u32 %0, %0, %0;" : "+r"(x));
#elif OP == 5  // LOP3 xor
            asm volatile("xor.b32 %0, %0, 1;" : "+r"(x));
#elif OP == 6  // PRMT
            asm volatile("prmt.b32 %0, %0, %0, 0x3210;" : "+r"(x));
#elif OP == 7  // FMNMX
            asm volatile("min.f32 %0, %0, %0;" : "+f"(f));
#elif OP == 8  // MUFU.EX2
            asm volatile("ex2.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 9  // MUFU.RSQ
            asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 10  // MUFU.RCP
            asm volatile("rcp.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 11  // F2FP UNPACK e4m3
            { unsigned int tmp;
              asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)x));
              x = tmp; }
#elif OP == 12  // F2FP PACK e4m3
            { unsigned short tmp;
              asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(tmp) : "r"(x));
              x = (unsigned int)tmp; }
#elif OP == 13  // HFMA2
            asm volatile("fma.rn.f16x2 %0, %0, %0, %0;" : "+r"(x));
#elif OP == 14  // u64 add
            asm volatile("add.u64 %0, %0, %0;" : "+l"(lx));
            x = (unsigned)lx;
#elif OP == 15  // SHFL.SYNC.BFLY
            asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1F, -1;" : "+r"(x));
#elif OP == 16  // redux.sync.min
            asm volatile("redux.sync.min.u32 %0, %0, -1;" : "+r"(x));
#elif OP == 17  // DFMA (fp64)
            { double d = __int_as_float(x);
              asm volatile("fma.rn.f64 %0, %0, %0, %0;" : "+d"(d));
              x = __float_as_int((float)d); }
#endif
        }
    }
    if ((int)x == seed || f == 1.23456789f || lx == (unsigned long long)seed)
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = x + __float_as_int(f) + (unsigned)lx;
}
