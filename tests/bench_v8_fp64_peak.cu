// V8: FP64 DFMA peak verification
// Theoretical: 1:64 ratio vs FP32 → 76.97 TFLOPS / 64 = 1.20 TFLOPS at 2032 MHz boost
// Same 2-source pattern as FFMA
#ifndef N_CHAINS
#define N_CHAINS 8
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(double* A, double* B, double* C, int ITERS, int seed, int u2) {
    double v[N_CHAINS], b[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (double)(threadIdx.x + k);
        b[k] = (double)(threadIdx.x * 2 + k);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("fma.rn.f64 %0, %0, %1, %0;" : "+d"(v[k]) : "d"(b[k]));
            }
        }
    }

    double acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc += v[k];
    if ((int)acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
