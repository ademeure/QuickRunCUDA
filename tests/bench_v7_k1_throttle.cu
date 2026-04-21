// V7 K1: Sustained throttle threshold — at what wattage does B300 throttle?
// Run extended FFMA workload, sample power AND clock during run
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS_OUTER, int seed, int u2) {
    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;
    float c0=a, c1=a, c2=a, c3=a, c4=a, c5=a, c6=a, c7=a;
    float k0=b*0.99f, k1=b*0.98f, k2=b*0.97f, k3=b*0.96f;
    float k4=b*0.95f, k5=b*0.94f, k6=b*0.93f, k7=b*0.92f;

    #pragma unroll 1
    for (int j = 0; j < ITERS_OUTER; j++) {
        #pragma unroll 64
        for (int i = 0; i < 65536; i++) {
            c0 = c0 * k0 + b;
            c1 = c1 * k1 + b;
            c2 = c2 * k2 + b;
            c3 = c3 * k3 + b;
            c4 = c4 * k4 + b;
            c5 = c5 * k5 + b;
            c6 = c6 * k6 + b;
            c7 = c7 * k7 + b;
        }
    }

    float sum = c0+c1+c2+c3+c4+c5+c6+c7;
    if (sum == 1.234567e-30f) C[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}
