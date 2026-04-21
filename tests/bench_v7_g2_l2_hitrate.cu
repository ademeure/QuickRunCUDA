// V7 G2: L2 hit_rate per access pattern
// MODE 0: sequential (best L2 prefetch)
// MODE 1: strided (cache line aligned)
// MODE 2: random (worst case, defeats prefetcher)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int MASK = (16 * 1024 * 1024) - 1;  // 256 MB / 16 B

    float4 sum = make_float4(0,0,0,0);
    unsigned int idx = gtid;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Sequential per thread
        idx = (gtid + i * 256) & MASK;
#elif MODE == 1
        // Strided 4 KB
        idx = (gtid + i * 256) & MASK;  // 256 stride per thread
#elif MODE == 2
        // Pseudo-random via XORSHIFT (defeats prefetcher)
        unsigned int x = (gtid + i + 1);
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        idx = x & MASK;
#endif
        float4 v = A[idx];
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }

    if (sum.x == 1.234567e-30f) C[gtid] = sum;
}
