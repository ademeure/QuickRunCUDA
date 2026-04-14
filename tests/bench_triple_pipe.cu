// Triple-pipe saturation: can one kernel drive uniform + alu + fma concurrently?
// If yes, total sm_inst should exceed 4.00 (dispatch cap for vector pipes).

#ifndef N_UNIFORM
#define N_UNIFORM 0
#endif
#ifndef N_ALU
#define N_ALU 0
#endif
#ifndef N_FFMA
#define N_FFMA 0
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#if N_UNIFORM > 0
    // Warp-uniform chain (derived from blockIdx + seed → URx regs)
    unsigned int u0 = blockIdx.x, u1 = blockIdx.x + 17, u2_ = (unsigned)seed, u3 = (unsigned)seed * 3;
#endif
#if N_ALU > 0
    // Per-lane data for alu (PRMT chain)
    unsigned int a[8];
    #pragma unroll
    for (int k=0;k<8;k++) a[k] = 0xDEADBEEFu ^ (threadIdx.x*131 + k*17);
#endif
#if N_FFMA > 0
    // FFMA scalar chain (dual-issues to fmaH+fmaL)
    float f[8];
    #pragma unroll
    for (int k=0;k<8;k++) f[k] = 1.0001f + 0.0001f*(threadIdx.x + k*23);
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if N_UNIFORM > 0
            #pragma unroll
            for (int k=0;k<N_UNIFORM;k++) {
                u0 = u0 * 3 + u1;
                u1 = u1 + 17;
                u2_ = u2_ * 5 + u3;
                u3 = u3 + 23;
            }
#endif
#if N_ALU > 0
            #pragma unroll
            for (int k=0;k<N_ALU;k++) {
                unsigned int nxt = a[(k+1) & 7];
                asm volatile("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(a[k]) : "r"(nxt));
            }
#endif
#if N_FFMA > 0
            #pragma unroll
            for (int k=0;k<N_FFMA;k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[k]) : "f"(1.000001f), "f"(0.9999f));
            }
#endif
        }
    }

    unsigned int acc = 0;
#if N_UNIFORM > 0
    acc ^= u0 + u1 + u2_ + u3;
#endif
#if N_ALU > 0
    #pragma unroll
    for (int k=0;k<8;k++) acc ^= a[k];
#endif
#if N_FFMA > 0
    #pragma unroll
    for (int k=0;k<8;k++) acc ^= __float_as_int(f[k]);
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
