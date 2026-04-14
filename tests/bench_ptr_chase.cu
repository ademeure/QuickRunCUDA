// Pointer-chase latency: single lane walks a cyclic perm over CHASE_SIZE bytes.

#ifndef CHASE_SIZE
#define CHASE_SIZE 4096
#endif
#ifndef UNROLL
#define UNROLL 8
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* arr = (unsigned int*)A;
    const int N = CHASE_SIZE / 4;

    // Thread 0 of block 0 initialises a Fibonacci-ish cycle (coprime with N for most sizes)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int stride = 17;   // skip 68 bytes → nearly-cacheline stride w/o perfect alignment
        for (int i = 0; i < N; i++) arr[i] = (unsigned int)((i + stride) % N);
    }
    __threadfence_system();
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned int idx = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            idx = arr[idx];
        }
    }
    if ((int)idx == seed) ((unsigned int*)C)[0] = idx;
}
