// HEAD-TO-HEAD: max-tuned LDG.E.128 read.
// Each thread issues 256-bit (8x u32) loads. Per-warp 1KB bursts.
// Address pattern walks WS_BYTES each iteration to control L2 vs DRAM regime.
// Anti-DCE: chain into accumulator and store under impossible if.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef UNROLL
#define UNROLL 32
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int ITERS, int seed, int WS_LOG2) {
    // WS_BYTES = 1 << WS_LOG2  (e.g. 26 = 64 MiB, 32 = 4 GiB)
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned warp_id = tid >> 5;
    unsigned lane = tid & 31;

    unsigned long long mask = (1ull << WS_LOG2) - 1ull;
    // Each warp owns a 1 KB burst per inner step; consecutive warps tile the address space
    // bytes_per_outer_step = gridDim.x * blockDim.x * 32  (every thread contributes 32 B)
    unsigned long long bytes_per_step = (unsigned long long)gridDim.x * BLOCK_SIZE * 32;

    unsigned acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            // Per-thread 32 B at offset = tid*32 + (i+u)*bytes_per_step
            unsigned long long off = ((unsigned long long)tid * 32
                                    + (unsigned long long)(i + u) * bytes_per_step) & mask;
            unsigned long long addr = (unsigned long long)A + off;
            unsigned x0,x1,x2,x3,x4,x5,x6,x7;
            asm volatile("ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3),"=r"(x4),"=r"(x5),"=r"(x6),"=r"(x7)
                : "l"(addr));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
            acc4^=x4; acc5^=x5; acc6^=x6; acc7^=x7;
        }
    }
    // Anti-DCE
    unsigned v = acc0^acc1^acc2^acc3^acc4^acc5^acc6^acc7;
    if ((int)v == seed) C[tid & 0xFFFF] = v;  // safe: writes to first 256 KB of C
}
