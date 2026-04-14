// Cross-pipe contention matrix: confirm independence claims.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef N_FFMA
#define N_FFMA 0
#endif
#ifndef N_OP
#define N_OP 0
#endif
#ifndef OP
#define OP 0      // 0=MUFU.EX2 (xu) 1=LDS (lsu) 2=ATOMS (lsu) 3=FMNMX (alu) 4=SHFL (lsu)
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#if N_FFMA > 0
    float f[N_FFMA];
    #pragma unroll
    for (int k=0;k<N_FFMA;k++) f[k] = 1.0001f + 0.0001f*(threadIdx.x + k*23);
#endif
#if N_OP > 0
    float g[N_OP];
    unsigned int u[N_OP];
    #pragma unroll
    for (int k=0;k<N_OP;k++) { g[k] = 1.0001f + 0.0001f*(k*17); u[k] = threadIdx.x + k*31; }
    if (threadIdx.x < BLOCK_SIZE*N_OP) smem[threadIdx.x] = threadIdx.x;
    __syncthreads();
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if N_FFMA > 0
            #pragma unroll
            for (int k=0;k<N_FFMA;k++) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[k]) : "f"(1.000001f), "f"(0.9999f));
#endif
#if N_OP > 0
            #pragma unroll
            for (int k=0;k<N_OP;k++) {
#if OP == 0  // MUFU.EX2
                asm volatile("ex2.approx.f32 %0, %0;" : "+f"(g[k]));
#elif OP == 1  // LDS
                unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x + k*BLOCK_SIZE]);
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(u[k]) : "r"(base));
#elif OP == 2  // ATOMS.ADD
                unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x + k*BLOCK_SIZE]);
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(u[k]) : "r"(base), "r"(1u));
#elif OP == 3  // FMNMX
                asm volatile("min.f32 %0, %0, %1;" : "+f"(g[k]) : "f"(1.5f));
#elif OP == 4  // SHFL
                asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1F, -1;" : "+r"(u[k]));
#endif
            }
#endif
        }
    }

    unsigned int acc = 0;
#if N_FFMA > 0
    #pragma unroll
    for (int k=0;k<N_FFMA;k++) acc ^= __float_as_int(f[k]);
#endif
#if N_OP > 0
    #pragma unroll
    for (int k=0;k<N_OP;k++) acc ^= u[k] ^ __float_as_int(g[k]);
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
