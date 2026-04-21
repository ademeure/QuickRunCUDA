// V8: FADD peak — expect 1/2 FFMA's FLOP rate (1 op vs 2 per inst)
#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef OP
#define OP 0  // 0=FADD, 1=FMUL, 2=FFMA
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float v[N_CHAINS], b[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (float)(threadIdx.x + k);
        b[k] = (float)(threadIdx.x * 2 + k);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if OP == 0
                asm volatile("add.f32 %0, %0, %1;" : "+f"(v[k]) : "f"(b[k]));
#elif OP == 1
                asm volatile("mul.f32 %0, %0, %1;" : "+f"(v[k]) : "f"(b[k]));
#elif OP == 2
                asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(v[k]) : "f"(b[k]));
#endif
            }
        }
    }

    float acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc += v[k];
    if ((int)acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
