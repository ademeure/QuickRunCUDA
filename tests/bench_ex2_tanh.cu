// ex2.approx / tanh.approx across all supported types.
// MODE=0: single-chain latency via clock64 bracketing.
// MODE=1: multi-chain throughput (full block, full grid).

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

#if MODE == 0
// ===== LATENCY MODE =====
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float f = 0.3141592f;
    unsigned int ux = 0x3C003C00u;  // bf16 / f16 pair initializer
    unsigned short us = 0x3C00;     // scalar half

    unsigned long long total_dt = 0;
    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0   // ex2.approx.f32
            asm volatile("ex2.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 1 // ex2.approx.ftz.f32
            asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f));
#elif OP == 2 // ex2.approx.f16
            asm volatile("ex2.approx.f16 %0, %0;" : "+h"(us));
#elif OP == 3 // ex2.approx.f16x2
            asm volatile("ex2.approx.f16x2 %0, %0;" : "+r"(ux));
#elif OP == 4 // ex2.approx.ftz.bf16
            asm volatile("ex2.approx.ftz.bf16 %0, %0;" : "+h"(us));
#elif OP == 5 // ex2.approx.ftz.bf16x2
            asm volatile("ex2.approx.ftz.bf16x2 %0, %0;" : "+r"(ux));
#elif OP == 10 // tanh.approx.f32
            asm volatile("tanh.approx.f32 %0, %0;" : "+f"(f));
#elif OP == 11 // tanh.approx.f16
            asm volatile("tanh.approx.f16 %0, %0;" : "+h"(us));
#elif OP == 12 // tanh.approx.f16x2
            asm volatile("tanh.approx.f16x2 %0, %0;" : "+r"(ux));
#elif OP == 13 // tanh.approx.bf16
            asm volatile("tanh.approx.bf16 %0, %0;" : "+h"(us));
#elif OP == 14 // tanh.approx.bf16x2
            asm volatile("tanh.approx.bf16x2 %0, %0;" : "+r"(ux));
#endif
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }
    if (__float_as_int(f) == seed || (int)ux == seed || (int)us == seed)
        ((float*)C)[0] = f + __int_as_float(ux) + (float)us;
    ((unsigned long long*)C)[1] = total_dt;
}
#else
// ===== THROUGHPUT MODE =====
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float ff[CHAINS];
    unsigned int uu[CHAINS];
    unsigned short uh[CHAINS];
    #pragma unroll
    for (int k=0;k<CHAINS;k++) { ff[k]=0.1f*(k+1); uu[k]=0x3C003C00u^(k<<4); uh[k]=0x3C00^k; }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<CHAINS;k++) {
#if OP == 0
                asm volatile("ex2.approx.f32 %0, %0;" : "+f"(ff[k]));
#elif OP == 1
                asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(ff[k]));
#elif OP == 2
                asm volatile("ex2.approx.f16 %0, %0;" : "+h"(uh[k]));
#elif OP == 3
                asm volatile("ex2.approx.f16x2 %0, %0;" : "+r"(uu[k]));
#elif OP == 4
                asm volatile("ex2.approx.ftz.bf16 %0, %0;" : "+h"(uh[k]));
#elif OP == 5
                asm volatile("ex2.approx.ftz.bf16x2 %0, %0;" : "+r"(uu[k]));
#elif OP == 10
                asm volatile("tanh.approx.f32 %0, %0;" : "+f"(ff[k]));
#elif OP == 11
                asm volatile("tanh.approx.f16 %0, %0;" : "+h"(uh[k]));
#elif OP == 12
                asm volatile("tanh.approx.f16x2 %0, %0;" : "+r"(uu[k]));
#elif OP == 13
                asm volatile("tanh.approx.bf16 %0, %0;" : "+h"(uh[k]));
#elif OP == 14
                asm volatile("tanh.approx.bf16x2 %0, %0;" : "+r"(uu[k]));
#endif
            }
        }
    }
    float acc = 0;
    #pragma unroll
    for (int k=0;k<CHAINS;k++) acc += ff[k] + __int_as_float(uu[k]) + (float)uh[k];
    if (__float_as_int(acc) == seed) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
#endif
