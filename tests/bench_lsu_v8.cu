// LSU v8 (256-bit) streaming load throughput per SM.
// One CTA, N warps, ITERS x UNROLL per thread.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef UNROLL
#define UNROLL 8
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = gridDim.x * blockDim.x;
    unsigned int acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,acc5=0,acc6=0,acc7=0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned long long off = (unsigned long long)tid * 32
                                   + (unsigned long long)(i+j) * 32 * total;
            off &= 0x3FFFFFFFull;
            unsigned int x0,x1,x2,x3,x4,x5,x6,x7;
            asm volatile("ld.global.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3),
                  "=r"(x4),"=r"(x5),"=r"(x6),"=r"(x7)
                : "l"((unsigned long long)A + off));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
            acc4^=x4; acc5^=x5; acc6^=x6; acc7^=x7;
        }
    }
    unsigned int r = acc0^acc1^acc2^acc3^acc4^acc5^acc6^acc7;
    if ((int)r == seed) ((unsigned int*)C)[tid] = r;
}
