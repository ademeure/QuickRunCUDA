// Contention test using PRMT for the "bitwise ALU op" — PRMT is a byte
// permute that runs on pipe_alu but CANNOT be fused into a 3-input LOP3
// (different instruction family). Each inline asm = exactly 1 SASS PRMT.
// Cross-chain dependency (l[k] depends on l[(k+1)&mask]) keeps chains alive
// and prevents any algebraic simplification.

#ifndef N_FFMA2
#define N_FFMA2 0
#endif
#ifndef N_UNPACK
#define N_UNPACK 0
#endif
#ifndef N_PRMT
#define N_PRMT 0
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

// Round N_PRMT up to power-of-2 at compile time for cheap mod via AND mask.
#if N_PRMT == 1 || N_PRMT == 2 || N_PRMT == 4 || N_PRMT == 8 || N_PRMT == 16 || N_PRMT == 32
#define PRMT_MASK (N_PRMT - 1)
#else
#define PRMT_MASK 0
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
#if N_UNPACK > 0
    unsigned int u[N_UNPACK];
    #pragma unroll
    for (int k = 0; k < N_UNPACK; k++) u[k] = 0x3C003C01u ^ (threadIdx.x * 137 + k * 23);
#endif
#if N_PRMT > 0
    unsigned int p[N_PRMT];
    #pragma unroll
    for (int k = 0; k < N_PRMT; k++) p[k] = 0xDEADBEEFu ^ (threadIdx.x * 131 + k * 17);
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
#if N_UNPACK > 0
            #pragma unroll
            for (int k = 0; k < N_UNPACK; k++) {
                unsigned int tmp;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)u[k]));
                u[k] = tmp;
            }
#endif
#if N_PRMT > 0
            // PRMT with cross-chain: each chain permutes its bytes using
            // another chain's current value as selector.
            // prmt.b32 dst, a, b, selector — picks 4 bytes from {a,b} per selector nibbles.
            // We use l[k]^l[(k+1)] as selector → true inter-chain data dep.
            #pragma unroll
            for (int k = 0; k < N_PRMT; k++) {
                unsigned int nxt = p[(k + 1) & PRMT_MASK];
                asm volatile("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(p[k]) : "r"(nxt));
            }
#endif
        }
    }

    unsigned long long acc = 0;
#if N_FFMA2 > 0
    #pragma unroll
    for (int k = 0; k < N_FFMA2; k++) acc ^= f[k];
#endif
#if N_UNPACK > 0
    #pragma unroll
    for (int k = 0; k < N_UNPACK; k++) acc ^= (unsigned long long)u[k];
#endif
#if N_PRMT > 0
    #pragma unroll
    for (int k = 0; k < N_PRMT; k++) acc ^= (unsigned long long)p[k];
#endif
    if (acc == (unsigned long long)seed) ((unsigned long long*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
