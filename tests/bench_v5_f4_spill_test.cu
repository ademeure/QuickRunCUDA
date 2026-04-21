// V5 F4: Register usage and spill detection via ncu
// MODE 0: light register pressure (< 32 regs)
// MODE 1: medium pressure (~ 64 regs)
// MODE 2: heavy pressure (forces spills via maxregcount or simply too many live vars)
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
#define LANES 4
#elif MODE == 1
#define LANES 16
#elif MODE == 2
#define LANES 64
#elif MODE == 3
// FORCE SPILL: 256 lanes with __launch_bounds__(256, 1) + maxregcount-style limit
#define LANES 256
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float v[LANES];
    #pragma unroll
    for (int i = 0; i < LANES; i++) {
        v[i] = (float)(threadIdx.x + i) * 0.001f;
    }

    float k = (float)blockIdx.x * 1.001f + 1.0f;

    #pragma unroll 1
    for (int j = 0; j < ITERS; j++) {
        #pragma unroll
        for (int i = 0; i < LANES; i++) {
            v[i] = v[i] * k + 0.001f;
        }
    }

    // Anti-DCE: must use all v[i] in reachable code
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < LANES; i++) sum += v[i];
    if (sum == 1.234567e-30f) C[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}
