#ifndef MODE
#define MODE 1
#endif
#ifndef K_INNER
#define K_INNER 64
#endif

#if MODE == 4
#define STRIDE_FL4 128        // 2 KB stride
#elif MODE == 5
#define STRIDE_FL4 2048       // 32 KB stride
#elif MODE == 6
#define STRIDE_FL4 65536      // 1 MB stride
#elif MODE == 7
#define STRIDE_FL4 1048576    // 16 MB stride
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int THREADS = gridDim.x * blockDim.x;
    long long T = THREADS;
    long long S = STRIDE_FL4;
    float4 sum = make_float4(0,0,0,0);
    // Only 1 iter + K_INNER loads per thread to keep footprint bounded
    #pragma unroll 8
    for (int k = 0; k < K_INNER; k++) {
        long long idx = (long long)k * T * S + (long long)gtid * S;
        // Wrap idx within buffer
        long long MASK = (1LL << 27) - 1;  // 2^27 float4 = 2 GB (128M float4)
        idx &= MASK;
        float4 v = A[idx];
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    if (sum.x == 1.234567e-30f) C[gtid] = sum;
}
