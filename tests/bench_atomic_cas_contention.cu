// atomicCAS contention scaling: how does throughput degrade with N
// contending threads on the same address?

#ifndef N_CONTEND
#define N_CONTEND 1
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int* loc = (int*)A;
    if (threadIdx.x == 0 && blockIdx.x == 0) loc[0] = 0;
    __syncthreads();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Only the first N_CONTEND threads participate
    if ((tid & 0xFF) >= N_CONTEND) return;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Pure CAS spin: keep trying until value matches
        int expected, desired;
        do {
            expected = atomicCAS(loc, 0, 0);   // probe
            desired = expected + 1;
        } while (atomicCAS(loc, expected, desired) != expected);
    }
}
