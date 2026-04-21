// V6 C3: Mixed FFMA + DRAM min-energy clock sweep
// Each thread: 4 FFMA per 1 LDG (compute + memory mixed)
#ifndef ITERS_INNER
#define ITERS_INNER 32768
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS_OUTER, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int MASK = 16 * 1024 * 1024 - 1;  // 256 MB buffer / 16 B

    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;
    float c0=a, c1=a, c2=a, c3=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;

    #pragma unroll 1
    for (int j = 0; j < ITERS_OUTER; j++) {
        #pragma unroll 32
        for (int i = 0; i < ITERS_INNER; i++) {
            // 1 LDG + 4 FFMA per inner iter
            unsigned int idx = (gtid + i * 37888 + j) & MASK;
            float4 v = ((float4*)A)[idx];
            c0 = c0 * k0 + v.x;
            c1 = c1 * k1 + v.y;
            c2 = c2 * k2 + v.z;
            c3 = c3 * k3 + v.w;
        }
    }

    float sum = c0+c1+c2+c3;
    if (sum == 1.234567e-30f) C[gtid] = sum;
}
