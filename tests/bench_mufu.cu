// MUFU (Special Function Unit) Throughput Microbenchmark
// Configurable chain count and type via -H defines
// MUFU_ASM: PTX instruction (default: ex2.approx.f32)
// N_CHAINS: number of independent chains (default: 4, max: 16)
// MUFU_PACKED: use "r" constraint for f16x2/bf16x2
// MUFU_HALF: use "h" constraint for f16/bf16

#ifndef MUFU_ASM
#define MUFU_ASM ex2.approx.f32
#endif
#ifndef N_CHAINS
#define N_CHAINS 4
#endif
#ifndef UNROLL
#define UNROLL 8
#endif

#define _S(x) #x
#define S(x) _S(x)

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int unused_1, int unused_2) {
    float tid_f = (float)(threadIdx.x & 0xFF) * 0.001f;

#if defined(MUFU_PACKED)
    // f16x2 / bf16x2: chains in 32-bit regs
    unsigned int r0  = 0x3C003C00u ^ threadIdx.x;
    unsigned int r1  = 0x3C013C01u ^ threadIdx.x;
    unsigned int r2  = 0x3C023C02u ^ threadIdx.x;
    unsigned int r3  = 0x3C033C03u ^ threadIdx.x;
    unsigned int r4  = 0x3C043C04u ^ threadIdx.x;
    unsigned int r5  = 0x3C053C05u ^ threadIdx.x;
    unsigned int r6  = 0x3C063C06u ^ threadIdx.x;
    unsigned int r7  = 0x3C073C07u ^ threadIdx.x;
    unsigned int r8  = 0x3C083C08u ^ threadIdx.x;
    unsigned int r9  = 0x3C093C09u ^ threadIdx.x;
    unsigned int r10 = 0x3C0A3C0Au ^ threadIdx.x;
    unsigned int r11 = 0x3C0B3C0Bu ^ threadIdx.x;
    unsigned int r12 = 0x3C0C3C0Cu ^ threadIdx.x;
    unsigned int r13 = 0x3C0D3C0Du ^ threadIdx.x;
    unsigned int r14 = 0x3C0E3C0Eu ^ threadIdx.x;
    unsigned int r15 = 0x3C0F3C0Fu ^ threadIdx.x;
    #define REG_C "+r"
    #define SINK_EXPR r0^r1^r2^r3^r4^r5^r6^r7^r8^r9^r10^r11^r12^r13^r14^r15

#elif defined(MUFU_HALF)
    // f16 / bf16: chains in 16-bit regs
    unsigned short r0  = 0x3C00 ^ (threadIdx.x & 0xFF);
    unsigned short r1  = 0x3C01 ^ (threadIdx.x & 0xFF);
    unsigned short r2  = 0x3C02 ^ (threadIdx.x & 0xFF);
    unsigned short r3  = 0x3C03 ^ (threadIdx.x & 0xFF);
    unsigned short r4  = 0x3C04 ^ (threadIdx.x & 0xFF);
    unsigned short r5  = 0x3C05 ^ (threadIdx.x & 0xFF);
    unsigned short r6  = 0x3C06 ^ (threadIdx.x & 0xFF);
    unsigned short r7  = 0x3C07 ^ (threadIdx.x & 0xFF);
    unsigned short r8  = 0x3C08 ^ (threadIdx.x & 0xFF);
    unsigned short r9  = 0x3C09 ^ (threadIdx.x & 0xFF);
    unsigned short r10 = 0x3C0A ^ (threadIdx.x & 0xFF);
    unsigned short r11 = 0x3C0B ^ (threadIdx.x & 0xFF);
    unsigned short r12 = 0x3C0C ^ (threadIdx.x & 0xFF);
    unsigned short r13 = 0x3C0D ^ (threadIdx.x & 0xFF);
    unsigned short r14 = 0x3C0E ^ (threadIdx.x & 0xFF);
    unsigned short r15 = 0x3C0F ^ (threadIdx.x & 0xFF);
    #define REG_C "+h"
    #define SINK_EXPR (unsigned int)(r0^r1^r2^r3^r4^r5^r6^r7^r8^r9^r10^r11^r12^r13^r14^r15)

#else
    // f32: chains in float regs
    float r0  = 0.50f + tid_f;
    float r1  = 0.51f + tid_f;
    float r2  = 0.52f + tid_f;
    float r3  = 0.53f + tid_f;
    float r4  = 0.54f + tid_f;
    float r5  = 0.55f + tid_f;
    float r6  = 0.56f + tid_f;
    float r7  = 0.57f + tid_f;
    float r8  = 0.58f + tid_f;
    float r9  = 0.59f + tid_f;
    float r10 = 0.60f + tid_f;
    float r11 = 0.61f + tid_f;
    float r12 = 0.62f + tid_f;
    float r13 = 0.63f + tid_f;
    float r14 = 0.64f + tid_f;
    float r15 = 0.65f + tid_f;
    #define REG_C "+f"
    #define SINK_EXPR __float_as_int(r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15)
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            asm volatile(
#if N_CHAINS >= 1
                S(MUFU_ASM) " %0, %0;\n\t"
#endif
#if N_CHAINS >= 2
                S(MUFU_ASM) " %1, %1;\n\t"
#endif
#if N_CHAINS >= 3
                S(MUFU_ASM) " %2, %2;\n\t"
#endif
#if N_CHAINS >= 4
                S(MUFU_ASM) " %3, %3;\n\t"
#endif
#if N_CHAINS >= 5
                S(MUFU_ASM) " %4, %4;\n\t"
#endif
#if N_CHAINS >= 6
                S(MUFU_ASM) " %5, %5;\n\t"
#endif
#if N_CHAINS >= 7
                S(MUFU_ASM) " %6, %6;\n\t"
#endif
#if N_CHAINS >= 8
                S(MUFU_ASM) " %7, %7;\n\t"
#endif
#if N_CHAINS >= 9
                S(MUFU_ASM) " %8, %8;\n\t"
#endif
#if N_CHAINS >= 10
                S(MUFU_ASM) " %9, %9;\n\t"
#endif
#if N_CHAINS >= 11
                S(MUFU_ASM) " %10, %10;\n\t"
#endif
#if N_CHAINS >= 12
                S(MUFU_ASM) " %11, %11;\n\t"
#endif
#if N_CHAINS >= 13
                S(MUFU_ASM) " %12, %12;\n\t"
#endif
#if N_CHAINS >= 14
                S(MUFU_ASM) " %13, %13;\n\t"
#endif
#if N_CHAINS >= 15
                S(MUFU_ASM) " %14, %14;\n\t"
#endif
#if N_CHAINS >= 16
                S(MUFU_ASM) " %15, %15;\n\t"
#endif
                :
#if N_CHAINS >= 1
                REG_C(r0)
#endif
#if N_CHAINS >= 2
                ,REG_C(r1)
#endif
#if N_CHAINS >= 3
                ,REG_C(r2)
#endif
#if N_CHAINS >= 4
                ,REG_C(r3)
#endif
#if N_CHAINS >= 5
                ,REG_C(r4)
#endif
#if N_CHAINS >= 6
                ,REG_C(r5)
#endif
#if N_CHAINS >= 7
                ,REG_C(r6)
#endif
#if N_CHAINS >= 8
                ,REG_C(r7)
#endif
#if N_CHAINS >= 9
                ,REG_C(r8)
#endif
#if N_CHAINS >= 10
                ,REG_C(r9)
#endif
#if N_CHAINS >= 11
                ,REG_C(r10)
#endif
#if N_CHAINS >= 12
                ,REG_C(r11)
#endif
#if N_CHAINS >= 13
                ,REG_C(r12)
#endif
#if N_CHAINS >= 14
                ,REG_C(r13)
#endif
#if N_CHAINS >= 15
                ,REG_C(r14)
#endif
#if N_CHAINS >= 16
                ,REG_C(r15)
#endif
            );
        }
    }

    if (threadIdx.x >= blockDim.x) {
        ((unsigned int*)C)[threadIdx.x] = SINK_EXPR;
    }
}
