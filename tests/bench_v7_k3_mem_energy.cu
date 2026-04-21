// V7 K3: Memory traffic vs energy scaling
// Run kernel that varies number of LDG ops per iteration
// Same outer iters, different memory intensity → energy/byte
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS_OUTER, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int MASK = (16 * 1024 * 1024) - 1;

    float4 sum = make_float4(0,0,0,0);

    #pragma unroll 1
    for (int j = 0; j < ITERS_OUTER; j++) {
        #pragma unroll 16
        for (int i = 0; i < 1024; i++) {
            unsigned int idx = (gtid * 1024 + i + j) & MASK;
            float4 v = A[idx];
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
    }

    if (sum.x == 1.234567e-30f) C[gtid] = sum;
}
