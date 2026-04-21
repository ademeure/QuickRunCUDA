// V7 G5: Cache replacement policy detection
// MODE 0: hot data + cold sweep (LRU should evict hot)
// MODE 1: hot data + repeated touches (LRU should KEEP hot)
// Measure L2 hit rate for "hot" portion across both modes
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // HOT region: first 16 MB (1M float4)
    unsigned int HOT_MASK = (1024 * 1024) - 1;
    // COLD region: 256 MB beyond hot (16M float4)
    unsigned int COLD_BASE = 1024 * 1024;
    unsigned int COLD_MASK = (16 * 1024 * 1024) - 1;

    float4 sum = make_float4(0,0,0,0);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Hot once, then sweep cold once per iter
        if ((i & 1) == 0) {
            float4 v = A[(gtid + i) & HOT_MASK];
            sum.x += v.x;
        } else {
            float4 v = A[COLD_BASE + ((gtid + i) & COLD_MASK)];
            sum.x += v.x;
        }
#elif MODE == 1
        // Always hit hot region (LRU should keep it hot)
        float4 v = A[(gtid + i) & HOT_MASK];
        sum.x += v.x;
#endif
    }

    if (sum.x == 1.234567e-30f) C[gtid] = sum;
}
