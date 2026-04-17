// membar.sys cost as function of N pending writes per iter (1 thread per SM)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef NWRITES
#define NWRITES 1
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (threadIdx.x != 0) return;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid * 32;

    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < 100; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        // NWRITES pending writes
        #pragma unroll
        for (int j = 0; j < NWRITES; j++) {
            ((volatile unsigned*)my_addr)[j] = i + seed + j;
        }
        asm volatile("membar.sys;");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    C[blockIdx.x] = total / 100;
}
