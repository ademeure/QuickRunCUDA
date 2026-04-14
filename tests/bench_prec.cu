// Precision modifier costs: .ftz, .sat, .approx, .relu on FP operations.

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

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<8;k++) {
#if OP == 0   // fma.rn.f32
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[k]) : "f"(1.000001f), "f"(0.9999f));
#elif OP == 1 // fma.rn.ftz.f32
                asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;" : "+f"(f[k]) : "f"(1.000001f), "f"(0.9999f));
#elif OP == 2 // fma.rn.sat.f32 (saturate to [0,1])
                asm volatile("fma.rn.sat.f32 %0, %0, %1, %2;" : "+f"(f[k]) : "f"(1.000001f), "f"(0.9999f));
#elif OP == 3 // fma.rn.ftz.sat.f32
                asm volatile("fma.rn.ftz.sat.f32 %0, %0, %1, %2;" : "+f"(f[k]) : "f"(1.000001f), "f"(0.9999f));
#elif OP == 4 // fma.rn.relu.f32 (new on Ampere+)
                asm volatile("fma.rn.relu.f32 %0, %0, %1, %2;" : "+f"(f[k]) : "f"(1.000001f), "f"(0.9999f));
#elif OP == 5 // mul with modifier
                asm volatile("mul.rn.ftz.f32 %0, %0, %1;" : "+f"(f[k]) : "f"(1.0000001f));
#elif OP == 6 // add with sat
                asm volatile("add.rn.sat.f32 %0, %0, %1;" : "+f"(f[k]) : "f"(0.0001f));
#elif OP == 7 // f16x2 fma with relu (Blackwell)
                unsigned int x = __float_as_int(f[k]);
                asm volatile("fma.rn.relu.f16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(0x3C003C00u), "r"(0x3C003C00u));
                f[k] = __int_as_float(x);
#elif OP == 8 // bf16x2 fma with relu
                unsigned int x = __float_as_int(f[k]);
                asm volatile("fma.rn.relu.bf16x2 %0, %0, %1, %2;" : "+r"(x) : "r"(0x3F803F80u), "r"(0x3F803F80u));
                f[k] = __int_as_float(x);
#endif
            }
        }
    }
    float acc = 0;
    #pragma unroll
    for (int k=0;k<8;k++) acc += f[k];
    if (__float_as_int(acc) == seed) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
