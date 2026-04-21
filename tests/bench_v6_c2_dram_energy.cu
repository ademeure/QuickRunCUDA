// V6 C2: DRAM-bound streaming kernel for energy sweep
// Each thread streams from A[gtid*N..(gtid+1)*N) and writes sum to C[gtid]
// Simple grid-stride read+write, anti-DCE via sum write
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS_OUTER, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    int N = 16384;  // float4 elements per thread per outer iter
    // Buffer is 256 MB / sizeof(float4) = 16M float4 entries
    // Mask = 16M - 1 wraps the index to stay in-bounds
    unsigned int MASK = 16 * 1024 * 1024 - 1;

    float4 sum = make_float4(0,0,0,0);

    #pragma unroll 1
    for (int j = 0; j < ITERS_OUTER; j++) {
        #pragma unroll 8
        for (int i = 0; i < N; i++) {
            unsigned int idx = (gtid + i * total + j) & MASK;
            float4 v = A[idx];
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
    }

    // Anti-DCE
    if (sum.x == 1.234567e-30f) C[gtid] = sum;
}
