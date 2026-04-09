// FP32 FMA Throughput Microbenchmark
// 8 independent FMA accumulator chains per thread, unrolled
// Usage: ./QuickRunCUDA tests/bench_fp32_fma.cu -p -t 256 -0 <ITERS> -T 100 -N <ITERS*8*2/1e12> -U "TFLOPS"

#ifndef UNROLL
#define UNROLL 8
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int unused_1, int unused_2) {
    // Thread-dependent init to prevent cross-thread constant folding
    float tid_f = __int_as_float(threadIdx.x | 0x3F800000u); // 1.0 + small perturbation
    float a = 1.0000001f;
    float b = 0.9999999f;
    float r0 = tid_f, r1 = tid_f + 1.0f, r2 = tid_f + 2.0f, r3 = tid_f + 3.0f;
    float r4 = tid_f + 4.0f, r5 = tid_f + 5.0f, r6 = tid_f + 6.0f, r7 = tid_f + 7.0f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            asm volatile(
                "fma.rn.f32 %0, %8, %9, %0;\n\t"
                "fma.rn.f32 %1, %8, %9, %1;\n\t"
                "fma.rn.f32 %2, %8, %9, %2;\n\t"
                "fma.rn.f32 %3, %8, %9, %3;\n\t"
                "fma.rn.f32 %4, %8, %9, %4;\n\t"
                "fma.rn.f32 %5, %8, %9, %5;\n\t"
                "fma.rn.f32 %6, %8, %9, %6;\n\t"
                "fma.rn.f32 %7, %8, %9, %7;\n\t"
                : "+f"(r0), "+f"(r1), "+f"(r2), "+f"(r3),
                  "+f"(r4), "+f"(r5), "+f"(r6), "+f"(r7)
                : "f"(a), "f"(b)
            );
        }
    }

    // Prevent DCE
    if (threadIdx.x >= blockDim.x) {
        C[threadIdx.x] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    }
}
