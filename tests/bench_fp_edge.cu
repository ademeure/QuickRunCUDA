// FP edge-case ops: testp, sqrt variants, f32 min/max with NaN propagation.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float f[8];
    #pragma unroll
    for (int k=0;k<8;k++) f[k] = 1.0001f + 0.0001f*(threadIdx.x + k*23);
    unsigned int v = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<8;k++) {
#if OP == 0  // testp.finite.f32
                unsigned int p;
                asm volatile("{.reg .pred q; testp.finite.f32 q, %1; selp.u32 %0, 1, 0, q;}" : "=r"(p) : "f"(f[k]));
                v ^= p; f[k] += 0.001f;
#elif OP == 1  // testp.subnormal.f32
                unsigned int p;
                asm volatile("{.reg .pred q; testp.subnormal.f32 q, %1; selp.u32 %0, 1, 0, q;}" : "=r"(p) : "f"(f[k]));
                v ^= p; f[k] += 0.001f;
#elif OP == 2  // testp.number.f32
                unsigned int p;
                asm volatile("{.reg .pred q; testp.number.f32 q, %1; selp.u32 %0, 1, 0, q;}" : "=r"(p) : "f"(f[k]));
                v ^= p; f[k] += 0.001f;
#elif OP == 3  // testp.notanumber.f32
                unsigned int p;
                asm volatile("{.reg .pred q; testp.notanumber.f32 q, %1; selp.u32 %0, 1, 0, q;}" : "=r"(p) : "f"(f[k]));
                v ^= p; f[k] += 0.001f;
#elif OP == 4  // testp.infinite.f32
                unsigned int p;
                asm volatile("{.reg .pred q; testp.infinite.f32 q, %1; selp.u32 %0, 1, 0, q;}" : "=r"(p) : "f"(f[k]));
                v ^= p; f[k] += 0.001f;
#elif OP == 5  // ex2.approx.ftz.f32
                asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f[k]));
#elif OP == 6  // fchk (FP range check)
                unsigned int p;
                asm volatile("{.reg .pred q; testp.finite.f32 q, %1; selp.u32 %0, 1, 0, q;}" : "=r"(p) : "f"(f[k]));
                v ^= p;
                f[k] = f[k] * 1.000001f;
#elif OP == 7  // copysign.f32 (compiler likely → LOP3)
                asm volatile("copysign.f32 %0, %1, %0;" : "+f"(f[k]) : "f"(-1.0f));
#elif OP == 8  // cos.approx.ftz.f32
                asm volatile("cos.approx.ftz.f32 %0, %0;" : "+f"(f[k]));
#elif OP == 9  // sin.approx.ftz.f32
                asm volatile("sin.approx.ftz.f32 %0, %0;" : "+f"(f[k]));
#endif
            }
        }
    }
    float acc = v;
    #pragma unroll
    for (int k=0;k<8;k++) acc += f[k];
    if (__float_as_int(acc) == seed) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
