// V10: Global atomicAdd throughput vs contention (parallel to SMEM test)
#ifndef CONTEND
#define CONTEND 32
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int my_addr = tid % CONTEND;

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
        atomicAdd(&A[my_addr], 1u);
    }
}
