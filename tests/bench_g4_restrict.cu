// G4: __restrict__ impact on schedule
// Same kernel with vs without __restrict__ on input pointers
// Compare runtime + SASS to see if compiler can reorder loads
// MODE 0: no restrict (compiler must assume aliasing)
// MODE 1: __restrict__ (compiler can reorder)

#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
__device__ __forceinline__ float compute(float* a, float* b, float* c, int idx) {
#else
__device__ __forceinline__ float compute(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float* __restrict__ c, int idx) {
#endif
    float x0 = a[idx];
    c[idx] = x0 + 1.0f;  // potential aliasing with a/b — no restrict means compiler can't reorder
    float x1 = b[idx];
    c[idx + 1] = x1 + 2.0f;
    float x2 = a[idx + 1];
    c[idx + 2] = x2 + 3.0f;
    float x3 = b[idx + 1];
    c[idx + 3] = x3 + 4.0f;
    float x4 = a[idx + 2];
    c[idx + 4] = x4 + 5.0f;
    float x5 = b[idx + 2];
    c[idx + 5] = x5 + 6.0f;
    float x6 = a[idx + 3];
    c[idx + 6] = x6 + 7.0f;
    float x7 = b[idx + 3];
    c[idx + 7] = x7 + 8.0f;
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int base = (gtid * 16) & ((1<<24) - 1);

    float sink = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        sink += compute(A, B, C, base + i*16 % 1024);
    }
    if ((int)sink == seed) C[blockIdx.x] = sink;
}
