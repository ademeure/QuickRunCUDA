// Check what SASS fence.sc.sys emits under different conditions
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef SINGLE_THREAD
#define SINGLE_THREAD 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
#if SINGLE_THREAD
    if (threadIdx.x != 0) return;
#endif
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid * 16;
    *(volatile unsigned*)my_addr = seed;
    asm volatile("fence.sc.sys;");
    C[tid] = 1;
}
