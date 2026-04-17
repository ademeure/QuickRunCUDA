// Wave-quantum analysis: does grid size near multiples of 148 (SM count) affect performance?
// Test: fixed work per CTA, vary grid size, measure ms.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Fixed work per thread — similar to FMA peak but simpler
    float v = __int_as_float(tid + 1);
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        v = v * 1.5f + v;
    }
    if (__float_as_int(v) == seed) C[tid] = v;
}
