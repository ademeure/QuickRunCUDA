// FFMA throughput vs occupancy
#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef N_BLOCKS_PER_SM
#define N_BLOCKS_PER_SM 1
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, N_BLOCKS_PER_SM)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float v[N_CHAINS], b[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (float)(threadIdx.x + k);
        b[k] = (float)(threadIdx.x * 2 + k);
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
}
