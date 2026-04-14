// LDG latency at L1 hit / L2 hit / DRAM — pointer-chase through different-size windows.

#ifndef N_OPS
#define N_OPS 64
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 32
#endif
#ifndef WINDOW_KB
#define WINDOW_KB 1
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Thread 0 of block 0 initializes a random-perm cycle
    unsigned int* arr = (unsigned int*)A;
    const int N = WINDOW_KB * 256;   // # u32 elements

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Simple cyclic: stride a prime; most N sizes will be coprime
        int stride = 131;
        for (int i = 0; i < N; i++) arr[i] = (unsigned)((i + stride) % N);
    }
    asm volatile("membar.gl;");
    __syncthreads();

    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int idx = 0;
    unsigned long long total_dt = 0;
    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
            idx = arr[idx];
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }
    if ((int)idx == seed) ((unsigned int*)C)[0] = idx;
    ((unsigned long long*)C)[1] = total_dt;
}
