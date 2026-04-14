// Measure contention between ops on SAME pipe vs different pipes concretely.
// Baseline rate for each solo, then pair at balanced load, compute u.

#ifndef N_A
#define N_A 0
#endif
#ifndef N_B
#define N_B 0
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
#ifndef OP_A
#define OP_A 0   // 0=LOP3 alu, 1=IMAD fmaH, 2=FFMA fma(H+L), 3=MUFU xu, 4=LDS lsu, 5=PRMT alu
#endif
#ifndef OP_B
#define OP_B 2
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int a[8]; float fa[8]; unsigned int ia[8];
    #pragma unroll
    for (int k=0;k<8;k++) { a[k]=threadIdx.x*131+k*17; ia[k]=threadIdx.x+k; fa[k]=1.0001f+0.0001f*(threadIdx.x+k*23); }
    unsigned int sbase = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x]);
    if (threadIdx.x < 512) smem[threadIdx.x] = threadIdx.x;
    __syncthreads();

#define DO_OP(K,OP_ID) \
    if (OP_ID == 0) asm volatile("lop3.b32 %0, %0, %1, %2, 0xE8;" : "+r"(a[K]) : "r"(a[(K+1)&7]), "r"(a[(K+2)&7])); \
    else if (OP_ID == 1) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(ia[K]) : "r"(7u), "r"(1u)); \
    else if (OP_ID == 2) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fa[K]) : "f"(1.000001f), "f"(0.9999f)); \
    else if (OP_ID == 3) { asm volatile("ex2.approx.f32 %0, %0;" : "+f"(fa[K])); } \
    else if (OP_ID == 4) { asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ia[K]) : "r"(sbase)); } \
    else if (OP_ID == 5) asm volatile("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(a[K]) : "r"(a[(K+1)&7]));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if N_A > 0
            #pragma unroll
            for (int k=0;k<N_A;k++) DO_OP(k&7, OP_A)
#endif
#if N_B > 0
            #pragma unroll
            for (int k=0;k<N_B;k++) DO_OP(k&7, OP_B)
#endif
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k=0;k<8;k++) acc ^= a[k] ^ ia[k] ^ __float_as_int(fa[k]);
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
