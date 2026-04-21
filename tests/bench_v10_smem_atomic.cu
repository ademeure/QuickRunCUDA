// V10: SMEM atomicAdd throughput vs contention
// CONTEND threads contend on same address. Sweep CONTEND.
#ifndef CONTEND
#define CONTEND 32   // 1=no contention, 32=full warp contend, 256=full block
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    __shared__ unsigned int counters[256];
    int tid = threadIdx.x;

    if (tid < 256) counters[tid] = 0;
    __syncthreads();

    // Each thread's address: thread tid → counters[tid % CONTEND]
    // CONTEND=1: all threads hit counters[0] (max contention)
    // CONTEND=256: all unique (no contention)
    int my_addr = tid % CONTEND;

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
        atomicAdd(&counters[my_addr], 1u);
    }

    __syncthreads();
    if (tid < 32) C[blockIdx.x * 32 + tid] = counters[tid];
}
