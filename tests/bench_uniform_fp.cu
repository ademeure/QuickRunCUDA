// Try harder to force UFFMA/UFADD/UFMUL emission.
// Key: value must be truly warp-invariant AND written to uniform reg.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(128, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Use blockIdx (warp-uniform) in FP operations.
    // Multiple chains of FP on blockIdx-derived values.
    float f0 = (float)blockIdx.x;
    float f1 = (float)(blockIdx.x + 1);
    float f2 = (float)(blockIdx.x + 2);
    float f3 = (float)(blockIdx.x + 3);
    float c = (float)seed * 1.000001f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // chained FFMA on uniform values
            f0 = f0 * c + f1;
            f1 = f1 * c + f2;
            f2 = f2 * c + f3;
            f3 = f3 * c + f0;
#elif OP == 1  // inline UFFMA if we can reference uniform reg explicitly
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f0) : "f"(c), "f"(f1));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f1) : "f"(c), "f"(f2));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f2) : "f"(c), "f"(f3));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f3) : "f"(c), "f"(f0));
#elif OP == 2  // FMNMX chain on uniform
            f0 = fminf(f0, c);
            f1 = fminf(f1, c + 0.1f);
            f2 = fminf(f2, c + 0.2f);
            f3 = fminf(f3, c + 0.3f);
#elif OP == 3  // test FSEL: conditional on uniform predicate
            f0 = (blockIdx.x & 1) ? f0 * c : f0 + c;
            f1 = (blockIdx.x & 2) ? f1 * c : f1 + c;
            f2 = (blockIdx.x & 4) ? f2 * c : f2 + c;
            f3 = (blockIdx.x & 8) ? f3 * c : f3 + c;
#endif
        }
    }
    // Deposit into thread-dependent location to keep f0..f3 live
    if (__float_as_int(f0) == seed)
        ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = f0 + f1 + f2 + f3;
}
