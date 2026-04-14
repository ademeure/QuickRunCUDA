// All MUFU × FP32/FP16/BF16 × ftz / non-ftz variants.
// MODE 0: latency. MODE 1: throughput.

#ifndef N_OPS
#define N_OPS 128
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 256
#endif
#ifndef OP
#define OP 0
#endif
#ifndef MODE
#define MODE 0
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef CHAINS
#define CHAINS 8
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#define ONE_OP(F32, F16_REG, PAIR_REG) \
  do { \
    if (OP == 0)  asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(F32)); \
    else if (OP == 1)  asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(F32)); \
    else if (OP == 2)  asm volatile("sqrt.approx.f32 %0, %0;" : "+f"(F32)); \
    else if (OP == 3)  asm volatile("sqrt.approx.ftz.f32 %0, %0;" : "+f"(F32)); \
    else if (OP == 4)  asm volatile("lg2.approx.f32 %0, %0;" : "+f"(F32)); \
    else if (OP == 5)  asm volatile("lg2.approx.ftz.f32 %0, %0;" : "+f"(F32)); \
    /* half-precision variants (ftz usually not applicable — denormals not in f16) */ \
    else if (OP == 10) asm volatile("rsqrt.approx.f16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 11) asm volatile("rsqrt.approx.ftz.f16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 12) asm volatile("rsqrt.approx.f16x2 %0, %0;" : "+r"(PAIR_REG)); \
    else if (OP == 13) asm volatile("rsqrt.approx.ftz.f16x2 %0, %0;" : "+r"(PAIR_REG)); \
    else if (OP == 14) asm volatile("rsqrt.approx.bf16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 15) asm volatile("rsqrt.approx.ftz.bf16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 16) asm volatile("rsqrt.approx.bf16x2 %0, %0;" : "+r"(PAIR_REG)); \
    else if (OP == 17) asm volatile("rsqrt.approx.ftz.bf16x2 %0, %0;" : "+r"(PAIR_REG)); \
    /* sqrt.approx.f16 family */ \
    else if (OP == 20) asm volatile("sqrt.approx.f16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 21) asm volatile("sqrt.approx.ftz.f16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 22) asm volatile("sqrt.approx.f16x2 %0, %0;" : "+r"(PAIR_REG)); \
    else if (OP == 23) asm volatile("sqrt.approx.ftz.f16x2 %0, %0;" : "+r"(PAIR_REG)); \
    else if (OP == 24) asm volatile("sqrt.approx.bf16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 25) asm volatile("sqrt.approx.ftz.bf16 %0, %0;" : "+h"(F16_REG)); \
    /* lg2.approx f16/bf16 */ \
    else if (OP == 30) asm volatile("lg2.approx.f16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 31) asm volatile("lg2.approx.ftz.f16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 32) asm volatile("lg2.approx.bf16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 33) asm volatile("lg2.approx.ftz.bf16 %0, %0;" : "+h"(F16_REG)); \
    /* rcp.approx f16/bf16 */ \
    else if (OP == 40) asm volatile("rcp.approx.f16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 41) asm volatile("rcp.approx.ftz.f16 %0, %0;" : "+h"(F16_REG)); \
    else if (OP == 42) asm volatile("rcp.approx.bf16 %0, %0;" : "+h"(F16_REG)); \
  } while (0)

#if MODE == 0
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float f = 1.2345f;
    unsigned short us = 0x3C10;
    unsigned int ux = 0x3C003C10u;
    unsigned long long total_dt = 0;
    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) ONE_OP(f, us, ux);
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }
    if (__float_as_int(f) == seed || (int)us == seed || (int)ux == seed) ((float*)C)[0] = f + (float)us + __int_as_float(ux);
    ((unsigned long long*)C)[1] = total_dt;
}
#else
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float f[CHAINS]; unsigned short us[CHAINS]; unsigned int ux[CHAINS];
    #pragma unroll
    for (int k=0;k<CHAINS;k++) { f[k]=1.2345f+0.01f*k; us[k]=0x3C10^k; ux[k]=0x3C003C10u^(k<<4); }
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<CHAINS;k++) ONE_OP(f[k], us[k], ux[k]);
        }
    }
    float acc = 0;
    #pragma unroll
    for (int k=0;k<CHAINS;k++) acc += f[k] + (float)us[k] + __int_as_float(ux[k]);
    if (__float_as_int(acc) == seed) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
#endif
