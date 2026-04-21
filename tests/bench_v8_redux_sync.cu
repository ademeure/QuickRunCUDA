// V8: redux.sync.min/max vs SHFL — prior V4 claimed 4× SHFL speedup
// Test redux.sync.add.u32
#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef OP
#define OP 0  // 0=redux.add, 1=shfl chain
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(int* A, int* B, int* C, int ITERS, int seed, int u2) {
    unsigned v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) v[k] = threadIdx.x + k + 1;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if OP == 0
                // redux.sync.add.u32 reduces across warp
                asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(v[k]));
#elif OP == 1
                // SHFL full reduction (need 5 rounds for 32 threads)
                v[k] += __shfl_xor_sync(0xFFFFFFFF, v[k], 1);
                v[k] += __shfl_xor_sync(0xFFFFFFFF, v[k], 2);
                v[k] += __shfl_xor_sync(0xFFFFFFFF, v[k], 4);
                v[k] += __shfl_xor_sync(0xFFFFFFFF, v[k], 8);
                v[k] += __shfl_xor_sync(0xFFFFFFFF, v[k], 16);
#endif
            }
        }
    }

    int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc += (int)v[k];
    if (acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
