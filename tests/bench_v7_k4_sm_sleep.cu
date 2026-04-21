// V7 K4: Per-SM power-gating test
// Run same kernel with N blocks (N from 1 to 148)
// Measure power; if HW power-gates unused SMs, P should scale with N
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS_OUTER, int seed, int u2) {
    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;
    float c0=a, c1=a, c2=a, c3=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;

    #pragma unroll 1
    for (int j = 0; j < ITERS_OUTER; j++) {
        #pragma unroll 64
        for (int i = 0; i < 65536; i++) {
            c0 = c0 * k0 + b;
            c1 = c1 * k1 + b;
            c2 = c2 * k2 + b;
            c3 = c3 * k3 + b;
        }
    }
    if (c0+c1+c2+c3 == 1.234567e-30f) C[blockIdx.x * blockDim.x + threadIdx.x] = c0;
}
