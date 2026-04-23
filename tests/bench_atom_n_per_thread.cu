// Per-thread N=2 test: each thread picks one of 2 addresses based on (threadIdx & 1)
// This forces WARP-level alternation between 2 addresses
#ifndef N_ADDR
#define N_ADDR 1
#endif
#ifndef OFFSET_BYTES
#define OFFSET_BYTES 4
#endif
#ifndef ITERS
#define ITERS 1000
#endif

extern "C" __global__ __launch_bounds__(128, 1)
void kernel(unsigned int* A, float* B, float* C, int seed, int u1, int u2) {
    unsigned int* base = A;
    // Each thread picks one of N_ADDR addresses based on threadIdx
    unsigned int slot = threadIdx.x % N_ADDR;
    unsigned int* myaddr = base + slot * (OFFSET_BYTES / 4);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        atomicAdd(myaddr, 1u);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[1024 + blockIdx.x] = t1 - t0;
    }
}
