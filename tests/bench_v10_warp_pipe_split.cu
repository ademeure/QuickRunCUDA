// V10: per-warp pipe split — does 2 warps in same SMSP stack pipes?
// 256 threads = 8 warps. Half do pure FFMA, half do pure IADD.
// If SMSP can dispatch FMA and ALU same cycle from different warps → 2× throughput
#ifndef N_CHAINS
#define N_CHAINS 8
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int warp = threadIdx.x / 32;
    bool fma_warp = (warp < 4);  // First 4 warps do FFMA, last 4 do IADD

    if (fma_warp) {
        float v[N_CHAINS], b[N_CHAINS];
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
            v[k] = (float)(threadIdx.x + k) * 0.001f;
            b[k] = (float)(threadIdx.x * 2 + k) * 0.001f;
        }
        #pragma unroll 1
        for (int i = 0; i < ITERS; i += 16) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                #pragma unroll
                for (int k = 0; k < N_CHAINS; k++) {
                    asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(v[k]) : "f"(b[k]));
                }
            }
        }
        float acc = 0;
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) acc += v[k];
        if ((int)acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    } else {
        int v[N_CHAINS], b[N_CHAINS];
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
            v[k] = threadIdx.x + k;
            b[k] = threadIdx.x * 2 + k;
        }
        #pragma unroll 1
        for (int i = 0; i < ITERS; i += 16) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                #pragma unroll
                for (int k = 0; k < N_CHAINS; k++) {
                    asm volatile("add.s32 %0, %0, %1;" : "+r"(v[k]) : "r"(b[k]));
                }
            }
        }
        int acc = 0;
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) acc += v[k];
        if (acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = (float)acc;
    }
}
