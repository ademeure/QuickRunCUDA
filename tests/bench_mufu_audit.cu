// MUFU (multi-function unit) audit: ex2, lg2, sin, cos, rcp, rsqrt, tanh.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef INNER
#define INNER 64
#endif
#ifndef OUTER
#define OUTER 100
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    float v0 = __int_as_float(tid + 1) * 1e-30f;
    float v1 = __int_as_float(tid + 2) * 1e-30f;
    float v2 = __int_as_float(tid + 3) * 1e-30f;
    float v3 = __int_as_float(tid + 4) * 1e-30f;
    float v4 = __int_as_float(tid + 5) * 1e-30f;
    float v5 = __int_as_float(tid + 6) * 1e-30f;
    float v6 = __int_as_float(tid + 7) * 1e-30f;
    float v7 = __int_as_float(tid + 8) * 1e-30f;

    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
#if OP == 0  // ex2
            v0 = exp2f(v0); v1 = exp2f(v1); v2 = exp2f(v2); v3 = exp2f(v3);
            v4 = exp2f(v4); v5 = exp2f(v5); v6 = exp2f(v6); v7 = exp2f(v7);
#elif OP == 1  // lg2
            v0 = log2f(v0+1); v1 = log2f(v1+1); v2 = log2f(v2+1); v3 = log2f(v3+1);
            v4 = log2f(v4+1); v5 = log2f(v5+1); v6 = log2f(v6+1); v7 = log2f(v7+1);
#elif OP == 2  // sin
            v0 = __sinf(v0); v1 = __sinf(v1); v2 = __sinf(v2); v3 = __sinf(v3);
            v4 = __sinf(v4); v5 = __sinf(v5); v6 = __sinf(v6); v7 = __sinf(v7);
#elif OP == 3  // cos
            v0 = __cosf(v0); v1 = __cosf(v1); v2 = __cosf(v2); v3 = __cosf(v3);
            v4 = __cosf(v4); v5 = __cosf(v5); v6 = __cosf(v6); v7 = __cosf(v7);
#elif OP == 4  // rcp
            v0 = 1.0f/(v0+1); v1 = 1.0f/(v1+1); v2 = 1.0f/(v2+1); v3 = 1.0f/(v3+1);
            v4 = 1.0f/(v4+1); v5 = 1.0f/(v5+1); v6 = 1.0f/(v6+1); v7 = 1.0f/(v7+1);
#elif OP == 5  // rsqrt
            v0 = rsqrtf(v0+1); v1 = rsqrtf(v1+1); v2 = rsqrtf(v2+1); v3 = rsqrtf(v3+1);
            v4 = rsqrtf(v4+1); v5 = rsqrtf(v5+1); v6 = rsqrtf(v6+1); v7 = rsqrtf(v7+1);
#elif OP == 6  // tanh
            v0 = tanhf(v0); v1 = tanhf(v1); v2 = tanhf(v2); v3 = tanhf(v3);
            v4 = tanhf(v4); v5 = tanhf(v5); v6 = tanhf(v6); v7 = tanhf(v7);
#elif OP == 7  // sqrt
            v0 = sqrtf(v0+1); v1 = sqrtf(v1+1); v2 = sqrtf(v2+1); v3 = sqrtf(v3+1);
            v4 = sqrtf(v4+1); v5 = sqrtf(v5+1); v6 = sqrtf(v6+1); v7 = sqrtf(v7+1);
#endif
        }
    }
    float sum = v0+v1+v2+v3+v4+v5+v6+v7;
    if (__float_as_int(sum) == seed) C[tid] = sum;
}
