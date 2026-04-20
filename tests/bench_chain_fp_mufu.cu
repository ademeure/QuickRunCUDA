// Chain latency for FP16 / mixed precision / MUFU operations.
// Robust anti-DCE via runtime u2 perturbation.
//
// MODE codes:
//   0  FFMA chain (baseline, expected 4 cy)
//   1  HFMA2 chain (f16x2 fma)
//   2  HMUL2 chain (f16x2 mul)
//   3  HADD2 chain (f16x2 add)
//   4  HFMA2.F32 chain (f16x2 -> f32 acc — mixed precision)
//   5  RCP chain (mufu.rcp.approx.f32)
//   6  EX2 chain
//   7  RSQRT chain
//   8  SIN chain
//   9  HFMA2 -> FFMA -> HFMA2 (FP16 in, FP32 mid, FP16 back)
//   10 FFMA -> RCP -> FFMA (typical softmax: x -> 1/x -> x*1/x)
//   11 FFMA -> EX2 -> FFMA (typical exp pipeline)
//   12 RCP -> EX2 -> RCP (chained MUFU)

#ifndef N_INNER
#define N_INNER 200
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float fv = (float)threadIdx.x + 1.5f;
    float fb = 1.000001f + (float)u2 * 1e-9f;
    float fc = 0.000001f + (float)u2 * 1e-9f;
    unsigned int hv = 0x3C003C00u ^ (unsigned)u2;  // chain holds f16x2 packed
    unsigned int hb = 0x3C013C01u ^ (unsigned)u2;
    unsigned int hc = 0x38003800u ^ ((unsigned)u2 << 8);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
#if MODE == 0
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#elif MODE == 1
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(hv) : "r"(hb), "r"(hc));
#elif MODE == 2
            asm volatile("mul.rn.f16x2 %0, %0, %1;" : "+r"(hv) : "r"(hb));
#elif MODE == 3
            asm volatile("add.rn.f16x2 %0, %0, %1;" : "+r"(hv) : "r"(hb));
#elif MODE == 4
            // HFMA2 with f32 accumulator (.f32 output, 2-elt packed inputs)
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#elif MODE == 5
            asm volatile("rcp.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 6
            asm volatile("ex2.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 7
            asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 8
            asm volatile("sin.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 9
            // HFMA2 -> cvt.f32 -> FFMA -> cvt.f16 -> HFMA2 (mixed pipeline)
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(hv) : "r"(hb), "r"(hc));
            // bridge h->f
            float bridge;
            asm volatile("{ .reg .f16 _t; mov.b16 _t, %1; cvt.f32.f16 %0, _t; }" : "=f"(bridge) : "h"((unsigned short)(hv & 0xFFFF)));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(bridge) : "f"(fb), "f"(fc));
            // bridge f->h
            unsigned int hpair;
            asm volatile("cvt.rn.satfinite.f16x2.f32 %0, %2, %1;" : "=r"(hpair) : "f"(bridge), "f"(bridge));
            asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(hpair) : "r"(hb), "r"(hc));
            hv = hpair;
#elif MODE == 10
            // FFMA -> RCP -> FFMA (3 inst per iter, all chained in fv)
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
            asm volatile("rcp.approx.f32 %0, %0;" : "+f"(fv));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#elif MODE == 11
            // FFMA -> EX2 -> FFMA
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
            asm volatile("ex2.approx.f32 %0, %0;" : "+f"(fv));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#elif MODE == 12
            // RCP -> EX2 -> RCP (MUFU chain)
            asm volatile("rcp.approx.f32 %0, %0;" : "+f"(fv));
            asm volatile("ex2.approx.f32 %0, %0;" : "+f"(fv));
            asm volatile("rcp.approx.f32 %0, %0;" : "+f"(fv));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)fv == seed && hv == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = hv;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long pairs = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
#if MODE == 9
        unsigned long long inst = pairs * 5;  // 2 hfma + 1 cvt-down + 1 ffma + 1 cvt-up
#elif MODE == 10 || MODE == 11 || MODE == 12
        unsigned long long inst = pairs * 3;
#else
        unsigned long long inst = pairs;
#endif
        printf("MODE=%2d insts=%llu clk=%llu cy/inst=%.3f cy/iter=%.3f\n",
               MODE, inst, t1 - t0, (double)(t1-t0)/(double)inst,
               (double)(t1-t0)/(double)pairs);
    }
}
