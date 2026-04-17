// FFMA peak: 1024 FFMAs in fully-unrolled inner loop, outer loop 100x with `#pragma unroll 1`
// 8 independent chains. Seed-predicated output to defeat DCE.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef INNER
#define INNER 128   // FFMAs per chain in inner loop. 8 chains × 128 = 1024 FFMAs/inner-iter
#endif
#ifndef OUTER
#define OUTER 100
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 8 independent chains, distinct initial values per thread
    float v0 = __int_as_float(tid + 1);
    float v1 = __int_as_float(tid + 2);
    float v2 = __int_as_float(tid + 3);
    float v3 = __int_as_float(tid + 4);
    float v4 = __int_as_float(tid + 5);
    float v5 = __int_as_float(tid + 6);
    float v6 = __int_as_float(tid + 7);
    float v7 = __int_as_float(tid + 8);
    float y  = __int_as_float(tid + 9) * 1.000001f;  // multiplier

    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            // 8 independent FMA chains — each v_k = v_k * y + v_k
            v0 = v0 * y + v0;
            v1 = v1 * y + v1;
            v2 = v2 * y + v2;
            v3 = v3 * y + v3;
            v4 = v4 * y + v4;
            v5 = v5 * y + v5;
            v6 = v6 * y + v6;
            v7 = v7 * y + v7;
        }
    }

    // Seed-predicated unconditional store — runtime opaque, defeats compile-time DCE
    float sum = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    if (__float_as_int(sum) == seed) {
        C[tid] = sum;
    }
}
