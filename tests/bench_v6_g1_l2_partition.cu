// V6 G1: L2 partition awareness via stride patterns
// B300 has 12 HBM stacks → 12 L2 partitions
// Test: read stride patterns that hit 1 vs 12 partitions; measure BW
// MODE 0: stride = 256 B (one cache line, cycles partitions naturally)
// MODE 1: stride = 4 KB (page-aligned, may concentrate)
// MODE 2: stride = 64 KB (super-aligned, likely concentrate)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;

#if MODE == 0
    int STRIDE_F4 = 256 / 16;  // 16 float4s = 1 cache line apart per thread
#elif MODE == 1
    int STRIDE_F4 = 4096 / 16; // 256 float4s = 4 KB apart
#elif MODE == 2
    int STRIDE_F4 = 65536 / 16; // 4096 float4s = 64 KB apart
#endif

    unsigned int MASK = (256 * 1024 * 1024 / 16) - 1;  // 256 MB / 16 B
    float4 sum = make_float4(0,0,0,0);

    for (int i = 0; i < ITERS; i++) {
        unsigned int idx = (gtid * STRIDE_F4 + i) & MASK;
        float4 v = A[idx];
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }

    if (sum.x == 1.234567e-30f) C[gtid] = sum;
}
