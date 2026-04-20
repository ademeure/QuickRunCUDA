// Universal RF port pressure test for fma-family.
// FMA_TYPE: 0=FFMA(F32), 1=IMAD(int), 2=HFMA2(half2), 3=DFMA(F64)
// PORT_MODE: 0=1 unique, 1=2 unique, 2=3 unique

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
#ifndef PORT_MODE
#define PORT_MODE 2
#endif
#ifndef FMA_TYPE
#define FMA_TYPE 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv[N_CHAINS], fb[N_CHAINS], fc[N_CHAINS];
    int   iv[N_CHAINS], ib[N_CHAINS], ic[N_CHAINS];
    unsigned hv[N_CHAINS], hb[N_CHAINS], hc[N_CHAINS];
    double dv[N_CHAINS], db[N_CHAINS], dc[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        fv[k] = (float)(threadIdx.x + k);
        fb[k] = (float)(threadIdx.x * 2 + k);
        fc[k] = (float)(threadIdx.x * 3 + k);
        iv[k] = (int)(threadIdx.x + k);
        ib[k] = (int)(threadIdx.x * 2 + k);
        ic[k] = (int)(threadIdx.x * 3 + k);
        hv[k] = 0xDEAD0000u + threadIdx.x + k;
        hb[k] = 0xBEEF0000u + threadIdx.x + k;
        hc[k] = 0xCAFE0000u + threadIdx.x + k;
        dv[k] = (double)(threadIdx.x + k);
        db[k] = (double)(threadIdx.x * 2 + k);
        dc[k] = (double)(threadIdx.x * 3 + k);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if FMA_TYPE == 0  // FFMA
  #if PORT_MODE == 0
                asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(fv[k]));
  #elif PORT_MODE == 1
                asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv[k]) : "f"(fb[k]));
  #else
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv[k]) : "f"(fb[k]), "f"(fc[k]));
  #endif
#elif FMA_TYPE == 1  // IMAD
  #if PORT_MODE == 0
                asm volatile("mad.lo.s32 %0, %0, %0, %0;" : "+r"(iv[k]));
  #elif PORT_MODE == 1
                asm volatile("mad.lo.s32 %0, %0, %1, %0;" : "+r"(iv[k]) : "r"(ib[k]));
  #else
                asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(iv[k]) : "r"(ib[k]), "r"(ic[k]));
  #endif
#elif FMA_TYPE == 2  // HFMA2 (FP16x2)
  #if PORT_MODE == 0
                asm volatile("fma.rn.f16x2 %0, %0, %0, %0;" : "+r"(hv[k]));
  #elif PORT_MODE == 1
                asm volatile("fma.rn.f16x2 %0, %0, %1, %0;" : "+r"(hv[k]) : "r"(hb[k]));
  #else
                asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(hv[k]) : "r"(hb[k]), "r"(hc[k]));
  #endif
#elif FMA_TYPE == 3  // DFMA (F64)
  #if PORT_MODE == 0
                asm volatile("fma.rn.f64 %0, %0, %0, %0;" : "+d"(dv[k]));
  #elif PORT_MODE == 1
                asm volatile("fma.rn.f64 %0, %0, %1, %0;" : "+d"(dv[k]) : "d"(db[k]));
  #else
                asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(dv[k]) : "d"(db[k]), "d"(dc[k]));
  #endif
#endif
            }
        }
    }

    float facc = 0.0f; int iacc = 0; unsigned hacc = 0; double dacc = 0.0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) { facc += fv[k]; iacc ^= iv[k]; hacc ^= hv[k]; dacc += dv[k]; }
    if (((int)facc == seed) && (iacc == seed) && ((int)hacc == seed) && ((int)dacc == seed))
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)facc;
}
