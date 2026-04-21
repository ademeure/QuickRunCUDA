// V8 I1 extended: hash-probe — does any power-of-2 stride cause stack concentration?
// Fixed total loads; stride varies. Report DRAM bw via ncu.
#ifndef STRIDE_FL4
#define STRIDE_FL4 1
#endif
#ifndef K_INNER
#define K_INNER 64
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int THREADS = gridDim.x * blockDim.x;
    long long T = THREADS;
    long long S = STRIDE_FL4;
    // Wrap index into 2 GB buffer (= 128M float4 = 2^27)
    // This could alias if stride × K × THREADS > buffer / 2
    long long MASK = (1LL << 27) - 1;

    float4 sum = make_float4(0,0,0,0);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        long long base = (long long)i * T * S;
        #pragma unroll 8
        for (int k = 0; k < K_INNER; k++) {
            long long idx = (base + (long long)k * T * S + (long long)gtid * S) & MASK;
            float4 v = A[idx];
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
    }

    if (sum.x == 1.234567e-30f) C[gtid] = sum;
}
