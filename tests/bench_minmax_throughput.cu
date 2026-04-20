// IMNMX / FMNMX / FMINMAX throughput.
// Modes:
//   0: IMNMX.S32 (signed int min/max - PTX min)
//   1: IMNMX.U32 (unsigned)
//   2: FMNMX.F32 (FP32 min/max - PTX min.f32)
//   3: FMNMX.F32 NaN-aware (PTX min.NaN.f32)
//   4: FFMINMAX.F16x2 (PTX min.f16x2)
//   5: FFMA-baseline for comparison

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP_MODE
#define OP_MODE 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int   iv[N_CHAINS], ib[N_CHAINS];
    unsigned uv[N_CHAINS], ub[N_CHAINS];
    float fv[N_CHAINS], fb[N_CHAINS];
    unsigned hv[N_CHAINS], hb[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        iv[k] = (int)(threadIdx.x * 131 + k * 17);
        ib[k] = (int)(threadIdx.x * 271 + k * 23);
        uv[k] = (unsigned)(threadIdx.x * 131 + k * 17);
        ub[k] = (unsigned)(threadIdx.x * 271 + k * 23);
        fv[k] = (float)(threadIdx.x + k);
        fb[k] = (float)(threadIdx.x * 2 + k);
        hv[k] = 0xDEAD0000u + (threadIdx.x + k);  // 2x f16 packed
        hb[k] = 0xBEEF0000u + (threadIdx.x + k);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if OP_MODE == 0
                asm volatile("min.s32 %0, %0, %1;" : "+r"(iv[k]) : "r"(ib[k]));
#elif OP_MODE == 1
                asm volatile("min.u32 %0, %0, %1;" : "+r"(uv[k]) : "r"(ub[k]));
#elif OP_MODE == 2
                asm volatile("min.f32 %0, %0, %1;" : "+f"(fv[k]) : "f"(fb[k]));
#elif OP_MODE == 3
                asm volatile("min.NaN.f32 %0, %0, %1;" : "+f"(fv[k]) : "f"(fb[k]));
#elif OP_MODE == 4
                asm volatile("min.f16x2 %0, %0, %1;" : "+r"(hv[k]) : "r"(hb[k]));
#elif OP_MODE == 5
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv[k]) : "f"(fb[k]), "f"(fb[k]));
#endif
            }
        }
    }

    int iacc = 0; unsigned uacc = 0; float facc = 0.0f; unsigned hacc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) { iacc ^= iv[k]; uacc ^= uv[k]; facc += fv[k]; hacc ^= hv[k]; }
    if ((iacc == seed) && ((int)uacc == seed) && ((int)facc == seed) && ((int)hacc == seed))
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = uacc;
}
