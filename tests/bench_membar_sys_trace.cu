// Per-iteration membar.sys timing trace. 1 thread per SM measures each iter.
// Records the (t1-t0) for every (write + membar.sys) cycle.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 100
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (threadIdx.x != 0) return;  // 1 thread per SM
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid;

    // Each SM writes its 100 latencies to C[blockIdx.x * 100 + iter]
    unsigned* out = C + blockIdx.x * ITERS;

    unsigned long long t0, t1;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        // Scalar write + membar.sys (in flight)
        *(volatile unsigned*)my_addr = i + seed;
        asm volatile("membar.sys;");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        out[i] = (unsigned)(t1 - t0);
    }
}
