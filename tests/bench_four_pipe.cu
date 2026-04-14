// Can we simultaneously saturate alu + fma + lsu + xu?
// Measure total SASS throughput achievable in one kernel.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef N_ALU
#define N_ALU 4
#endif
#ifndef N_FMA
#define N_FMA 4
#endif
#ifndef N_LSU
#define N_LSU 4
#endif
#ifndef N_XU
#define N_XU 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int a[8], u[8]; float f[8], g[8];
    #pragma unroll
    for (int k=0;k<8;k++) { a[k]=threadIdx.x*131+k*17; u[k]=threadIdx.x; f[k]=1.0001f+0.0001f*(threadIdx.x+k*23); g[k]=1.0001f+0.0001f*(k*13); }
    unsigned int sbase = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x]);
    if (threadIdx.x < 1024) smem[threadIdx.x] = threadIdx.x;
    __syncthreads();

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if N_ALU > 0
            #pragma unroll
            for (int k=0;k<N_ALU;k++) asm volatile("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(a[k&7]) : "r"(a[(k+1)&7]));
#endif
#if N_FMA > 0
            #pragma unroll
            for (int k=0;k<N_FMA;k++) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[k&7]) : "f"(1.000001f), "f"(0.9999f));
#endif
#if N_LSU > 0
            #pragma unroll
            for (int k=0;k<N_LSU;k++) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(u[k&7]) : "r"(sbase));
#endif
#if N_XU > 0
            #pragma unroll
            for (int k=0;k<N_XU;k++) asm volatile("ex2.approx.f32 %0, %0;" : "+f"(g[k&7]));
#endif
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k=0;k<8;k++) acc ^= a[k] ^ u[k] ^ __float_as_int(f[k]) ^ __float_as_int(g[k]);
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
