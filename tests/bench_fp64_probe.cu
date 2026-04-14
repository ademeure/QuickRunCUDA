// Quick FP64 and DMMA probe.
#ifndef OP
#define OP 0
#endif
extern "C" __global__ __launch_bounds__(256, 2)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    double f[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) f[k] = 1.0001 + 0.0001*(threadIdx.x + k*23);
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
#if OP == 0  // DFMA
                asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(f[k]) : "d"(1.000001), "d"(0.9999));
#elif OP == 1 // DADD
                asm volatile("add.rn.f64 %0, %0, %1;" : "+d"(f[k]) : "d"(0.000001));
#elif OP == 2 // DMUL
                asm volatile("mul.rn.f64 %0, %0, %1;" : "+d"(f[k]) : "d"(1.000001));
#endif
            }
        }
    }
    double acc = 0;
    #pragma unroll
    for (int k = 0; k < 4; k++) acc += f[k];
    if (acc == 0.0) ((double*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
