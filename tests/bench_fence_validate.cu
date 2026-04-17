// Validate fence cost with various pending-write scenarios + thread counts.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 1024
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned acc = seed;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0
        // membar.gl with NO pending writes
        asm volatile("membar.gl;");
#elif OP == 1
        // membar.gl WITH 1 pending write
        *(volatile unsigned*)my_addr = i;
        asm volatile("membar.gl;");
#elif OP == 2
        // membar.gl WITH 4 pending writes
        ((volatile unsigned*)my_addr)[0] = i;
        ((volatile unsigned*)my_addr)[1] = i+1;
        ((volatile unsigned*)my_addr)[2] = i+2;
        ((volatile unsigned*)my_addr)[3] = i+3;
        asm volatile("membar.gl;");
#elif OP == 3
        // membar.gl WITH 16 pending writes
        for (int j = 0; j < 16; j++) ((volatile unsigned*)my_addr)[j] = i + j;
        asm volatile("membar.gl;");
#elif OP == 4
        // fence.sc.gpu with no pending writes
        asm volatile("fence.sc.gpu;");
#elif OP == 5
        // fence.sc.gpu WITH 4 pending writes
        ((volatile unsigned*)my_addr)[0] = i;
        ((volatile unsigned*)my_addr)[1] = i+1;
        ((volatile unsigned*)my_addr)[2] = i+2;
        ((volatile unsigned*)my_addr)[3] = i+3;
        asm volatile("fence.sc.gpu;");
#elif OP == 6
        // membar.cta WITH 4 pending writes (CTA scope)
        ((volatile unsigned*)my_addr)[0] = i;
        ((volatile unsigned*)my_addr)[1] = i+1;
        ((volatile unsigned*)my_addr)[2] = i+2;
        ((volatile unsigned*)my_addr)[3] = i+3;
        asm volatile("membar.cta;");
#elif OP == 7
        // membar.sys WITH 4 pending writes
        ((volatile unsigned*)my_addr)[0] = i;
        ((volatile unsigned*)my_addr)[1] = i+1;
        ((volatile unsigned*)my_addr)[2] = i+2;
        ((volatile unsigned*)my_addr)[3] = i+3;
        asm volatile("membar.sys;");
#endif
        acc ^= i;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
    }
}
