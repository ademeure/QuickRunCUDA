// V8 I2: HBM stack contention — concentrate vs distribute access
// MODE 0: all threads hit same 256 MB region (likely 1-2 HBM stacks)
// MODE 1: threads stride across 12 GB (spread across all 12 stacks)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

#if MODE == 0
    // Concentrate: 256 MB buffer / 16 B = 16M float4
    unsigned int MASK = (16 * 1024 * 1024) - 1;
    unsigned int base = gtid & MASK;
#elif MODE == 1
    // Distribute: 12 GB / 16 B = 768M float4 entries — spread across HBM stacks
    unsigned int MASK = (768 * 1024 * 1024) - 1;
    unsigned int base = gtid & MASK;
#endif

    float4 sum = make_float4(0,0,0,0);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned int idx = (base + i * 37888) & MASK;
        float4 v = A[idx];
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }

    if (sum.x == 1.234567e-30f) C[gtid] = sum;
}
