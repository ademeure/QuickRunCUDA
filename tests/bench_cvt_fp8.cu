extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned acc = 0;
    #pragma unroll 1
    for (int i = 0; i < 1024; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            float f1 = __int_as_float(tid + i + j);
            float f2 = __int_as_float(tid + i + j + 1);
            unsigned out;
            // Try u16 short result (cvt.rn.satfinite.e4m3x2.f32 outputs 16-bit)
            unsigned short out16;
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                : "=h"(out16) : "f"(f1), "f"(f2));
            acc ^= out16;
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
    }
}
