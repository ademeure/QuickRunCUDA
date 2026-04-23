// Atomic contention sweep: N distinct addresses, 148 CTAs distributed round-robin.
// Measures cy/atomic for various N.
//
// -H "#define N_ADDR <n>"     -- number of distinct atomic addresses
// -H "#define OFFSET_BYTES <n>" -- byte offset between consecutive addresses
// -H "#define ITERS <n>"       -- atomicAdds per thread

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
    // N_ADDR addresses, each at base + i*OFFSET_BYTES
    unsigned int slot = blockIdx.x % N_ADDR;
    unsigned int* myaddr = base + slot * (OFFSET_BYTES / 4);

    unsigned int acc = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        acc += atomicAdd(myaddr, 1u);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (acc == 0xDEADBEEF) C[blockIdx.x] = (float)acc;
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[1024 + blockIdx.x] = t1 - t0;
    }
}
