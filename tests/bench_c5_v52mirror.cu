// Exact mirror of bench_ffma2_vs_f2fp.cu but parameterized by ALU_OP:
//   0 = F2FP UNPACK (catalog 0.84)
//   1 = PRMT (catalog 0.96 if u=1.95)
//   2 = LOP3 (catalog free)
//   3 = IADD3
// All other test infrastructure identical.
#ifndef N_FFMA2
#define N_FFMA2 0
#endif
#ifndef N_ALU
#define N_ALU 0
#endif
#ifndef ALU_OP
#define ALU_OP 0
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

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#if N_FFMA2 > 0
    unsigned long long f[N_FFMA2];
    #pragma unroll
    for (int k = 0; k < N_FFMA2; k++) {
        unsigned int ulo = __float_as_int(1.0001f + 0.0001f*(threadIdx.x + k*23));
        unsigned int uhi = __float_as_int(1.0002f + 0.0001f*(threadIdx.x + k*29));
        f[k] = ((unsigned long long)uhi << 32) | ulo;
    }
    unsigned int c1_u = __float_as_int(1.000001f);
    unsigned int c0_u = __float_as_int(0.9999f);
    unsigned long long c1 = ((unsigned long long)c1_u << 32) | c1_u;
    unsigned long long c0 = ((unsigned long long)c0_u << 32) | c0_u;
#endif
#if N_ALU > 0
    unsigned int u[N_ALU];
    unsigned int v[N_ALU];
    #pragma unroll
    for (int k = 0; k < N_ALU; k++) {
        u[k] = 0x3C003C01u ^ (threadIdx.x * 137 + k * 23);
        v[k] = 0xCAFEBABEu ^ (threadIdx.x * 71  + k * 19);
    }
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if N_FFMA2 > 0
            #pragma unroll
            for (int k = 0; k < N_FFMA2; k++) {
                asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(f[k]) : "l"(c1), "l"(c0));
            }
#endif
#if N_ALU > 0
            #pragma unroll
            for (int k = 0; k < N_ALU; k++) {
                unsigned int tmp;
#if ALU_OP == 0
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)u[k]));
                u[k] = tmp;
#elif ALU_OP == 1
                asm volatile("prmt.b32 %0, %0, %1, 0x7531;" : "+r"(u[k]) : "r"(v[k]));
#elif ALU_OP == 2
                asm volatile("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(u[k]) : "r"(v[k]), "r"(u[(k+1)%N_ALU]));
#elif ALU_OP == 3
                asm volatile("add.u32 %0, %0, %1;" : "+r"(u[k]) : "r"(v[k]));
#elif ALU_OP == 4
                // PRMT with single source (collapse port pressure)
                asm volatile("prmt.b32 %0, %0, %1, 0x7531;" : "+r"(u[k]) : "r"(u[k]));
#elif ALU_OP == 5
                // F2FP UNPACK with destination = same source register (force RMW)
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(u[k]) : "h"((unsigned short)u[k]));
#endif
            }
#endif
        }
    }

    unsigned long long acc = 0;
#if N_FFMA2 > 0
    #pragma unroll
    for (int k = 0; k < N_FFMA2; k++) acc ^= f[k];
#endif
#if N_ALU > 0
    #pragma unroll
    for (int k = 0; k < N_ALU; k++) acc ^= (unsigned long long)u[k];
#endif
    if (acc == (unsigned long long)seed)
        ((unsigned long long*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
