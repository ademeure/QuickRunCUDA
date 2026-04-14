// Maximum achievable work per cycle — combining multiple pipes at once with
// deep ILP. Goal: sustain > 4.00 sm_inst via pipe_uniform + vector pipes?

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef NF
#define NF 8
#endif
#ifndef NA
#define NA 8
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float f[NF];
    unsigned int a[NA];
    #pragma unroll
    for (int k=0;k<NF;k++) f[k] = 1.0001f + 0.0001f*(threadIdx.x+k*23);
    #pragma unroll
    for (int k=0;k<NA;k++) a[k] = threadIdx.x*131 + k*17;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Scalar FFMA (fma dual-pipe H+L) — fills 2 fma slots
            #pragma unroll
            for (int k=0;k<NF;k++)
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[k]) : "f"(1.000001f), "f"(0.9999f));
            // PRMT alu chain — fills alu
            #pragma unroll
            for (int k=0;k<NA;k++)
                asm volatile("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(a[k]) : "r"(a[(k+1)&(NA-1)]));
        }
    }
    float acc = 0;
    #pragma unroll
    for (int k=0;k<NF;k++) acc += f[k];
    #pragma unroll
    for (int k=0;k<NA;k++) acc += (float)a[k];
    if (__float_as_int(acc) == seed) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
