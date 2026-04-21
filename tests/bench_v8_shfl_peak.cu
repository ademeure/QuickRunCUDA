// V8: __shfl_sync peak — warp-level intrinsic
// Expected: 1 SHFL per cycle per SMSP = same rate as FFMA
#ifndef N_CHAINS
#define N_CHAINS 8
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(int* A, int* B, int* C, int ITERS, int seed, int u2) {
    int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) v[k] = threadIdx.x + k;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                // shuffle: each thread gets value from lane (tid + 1) & 31
                v[k] = __shfl_xor_sync(0xFFFFFFFF, v[k], 1);
            }
        }
    }

    int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc += v[k];
    if (acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
