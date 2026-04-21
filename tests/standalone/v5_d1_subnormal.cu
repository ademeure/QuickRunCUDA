// V5 D1: subnormal FFMA penalty WITHOUT -use_fast_math
// Build with: nvcc -arch=compute_103a -code=sm_103a -O3 -ftz=false -prec-div=true -prec-sqrt=true
//   (NO -use_fast_math)
#include <cuda_runtime.h>
#include <cstdio>

#ifndef MODE
#define MODE 0
#endif

__global__ __launch_bounds__(32, 1)
void kernel(float* out, int iters, int u2) {
    float x, y, z;
#if MODE == 0
    // Normal range
    x = 1.0f + (float)u2 * 1e-9f;
    y = 1.0001f;
    z = 0.5f;
#elif MODE == 1
    // Subnormal inputs via bit cast (~3.6e-43)
    unsigned int xi = 0x00000100u ^ (unsigned)u2;
    unsigned int yi = 0x00000200u ^ (unsigned)u2;
    unsigned int zi = 0x00000300u;
    x = __int_as_float(xi);
    y = __int_as_float(yi);
    z = __int_as_float(zi);
#elif MODE == 2
    // Operands that produce subnormal output (small * small = underflow)
    unsigned int xi = 0x18000001u ^ (unsigned)u2;
    unsigned int yi = 0x18000002u ^ (unsigned)u2;
    unsigned int zi = 0x00000001u;
    x = __int_as_float(xi);
    y = __int_as_float(yi);
    z = __int_as_float(zi);
#endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < iters; i++) {
        // 16 chained FFMA
        x = x*y + z;  x = x*y + z;  x = x*y + z;  x = x*y + z;
        x = x*y + z;  x = x*y + z;  x = x*y + z;  x = x*y + z;
        x = x*y + z;  x = x*y + z;  x = x*y + z;  x = x*y + z;
        x = x*y + z;  x = x*y + z;  x = x*y + z;  x = x*y + z;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.2f cy/fma=%.4f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)iters, (double)(t1-t0)/(double)iters/16.0);
    }
    if ((int)(x*1e30f) == 12345) out[0] = x;
}

int main() {
    float* d; cudaMalloc(&d, 1024);
    kernel<<<1, 32>>>(d, 100000, 7);
    cudaDeviceSynchronize();
    return 0;
}
