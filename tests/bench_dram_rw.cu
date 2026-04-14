// DRAM read vs write vs copy bandwidth (each stressed in isolation).

#ifndef UNROLL
#define UNROLL 4
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;
    unsigned int acc0=0,acc1=0,acc2=0,acc3=0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Sequential coalesced, guaranteed DRAM-size
            unsigned long long off = (unsigned long long)tid * 16 +
                                     (unsigned long long)(i + j) * 16 * total_threads;
            off &= 0x3FFFFFFFull;  // 1 GB wrap
            unsigned long long addr_a = (unsigned long long)A + off;
            unsigned long long addr_b = (unsigned long long)B + off;

#if MODE == 0  // READ only
            unsigned int x0,x1,x2,x3;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr_a));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
#elif MODE == 1  // WRITE only
            asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                :: "l"(addr_b),
                   "r"(tid+j),"r"(tid+j+1),"r"(tid+j+2),"r"(tid+j+3));
#elif MODE == 2  // READ + WRITE (copy)
            unsigned int x0,x1,x2,x3;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr_a));
            asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                :: "l"(addr_b), "r"(x0),"r"(x1),"r"(x2),"r"(x3));
            acc0^=x0;
#elif MODE == 3  // WRITE with .wb cache hint (keep in L2)
            asm volatile("st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};"
                :: "l"(addr_b), "r"(tid+j),"r"(tid+j+1),"r"(tid+j+2),"r"(tid+j+3));
#elif MODE == 4  // WRITE with .cs (streaming, evict L2)
            asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                :: "l"(addr_b), "r"(tid+j),"r"(tid+j+1),"r"(tid+j+2),"r"(tid+j+3));
#endif
        }
    }
    unsigned int r = acc0^acc1^acc2^acc3;
    if ((int)r == seed) ((unsigned int*)C)[tid] = r;
}
