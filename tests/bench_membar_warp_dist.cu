#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef OP
#define OP 0  // 0=8 warps x 1 write; 1=1 warp x 8 writes; 2=2 warps x 4 writes each; 3=4 warps x 2
#endif
#ifndef ITERS
#define ITERS 200
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned warp_id = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    unsigned* my_addr = A + tid * 16;

    // Determine NWRITES and ACTIVE_WARPS based on OP
#if OP == 0
    int active_warps = 8; int nw = 1;
#elif OP == 1
    int active_warps = 1; int nw = 8;
#elif OP == 2
    int active_warps = 2; int nw = 4;
#elif OP == 3
    int active_warps = 4; int nw = 2;
#endif
    // total writes per CTA = 32 (constant)
    if (warp_id >= active_warps) return;

    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        for (int j = 0; j < nw; j++) ((volatile unsigned*)my_addr)[j] = i + seed + j;
        asm volatile("fence.sc.sys;");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    if (lane == 0) C[blockIdx.x * active_warps + warp_id] = total / ITERS;
}
