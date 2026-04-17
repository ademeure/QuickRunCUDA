// F2FP throughput: audit FP-format conversion variants.
// SASS forms: F2FP.F16.E4M3, F2FP.F16.E5M2, F2FP.F16, F2FP.BF16, etc.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef ITERS
#define ITERS 1024
#endif
#ifndef OP
#define OP 0
#endif

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned in = (tid * 0x9E3779B9u) ^ (i + j) ^ seed;
            unsigned out;
#if OP == 0
            // FP32 → E4M3 (pack 2 FP32 to 2× E4M3 bytes)
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                : "=r"(out) : "f"(__int_as_float(in)), "f"(__int_as_float(in+1)));
#elif OP == 1
            // FP32 → E5M2
            asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
                : "=r"(out) : "f"(__int_as_float(in)), "f"(__int_as_float(in+1)));
#elif OP == 2
            // FP32 → FP16 (FFMA pipe)
            asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;"
                : "=r"(out) : "f"(__int_as_float(in)), "f"(__int_as_float(in+1)));
#elif OP == 3
            // FP32 → BF16
            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                : "=r"(out) : "f"(__int_as_float(in)), "f"(__int_as_float(in+1)));
#elif OP == 4
            // E4M3 → FP16 (UNPACK)
            unsigned o0, o1;
            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;"
                : "=r"(o0) : "h"((unsigned short)(in & 0xFFFF)));
            out = o0;
#elif OP == 5
            // FP16 → FP32
            float f;
            asm volatile("cvt.f32.f16 %0, %1;"
                : "=f"(f) : "h"((unsigned short)(in & 0xFFFF)));
            out = __float_as_int(f);
#endif
            acc ^= out;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
    }
}
