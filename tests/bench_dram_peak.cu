#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef OP
#define OP 0
#endif
#if OP == 1 || OP == 3 || OP == 5
#define BPI 32
#else
#define BPI 16
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int ITERS, int seed, int WS_BYTES) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = (unsigned)(WS_BYTES - 1);
    unsigned acc0=0, acc1=0, acc2=0, acc3=0, acc4=0, acc5=0, acc6=0, acc7=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned off = ((tid * BPI + (i + j) * BPI * gridDim.x * blockDim.x) & mask);
            off = off & ~(BPI - 1u);  // align to BPI
            unsigned long long addr_a = (unsigned long long)A + off;
            unsigned long long addr_b = (unsigned long long)B + off;
#if OP == 0
            unsigned x0,x1,x2,x3;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr_a));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
#elif OP == 1
            unsigned x0,x1,x2,x3,x4,x5,x6,x7;
            asm volatile("ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3),"=r"(x4),"=r"(x5),"=r"(x6),"=r"(x7) : "l"(addr_a));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
            acc4^=x4; acc5^=x5; acc6^=x6; acc7^=x7;
#elif OP == 2
            unsigned x0,x1,x2,x3;
            asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr_a));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
#elif OP == 4
            asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                :: "l"(addr_b), "r"(tid+i+j),"r"(tid+i+j+1),"r"(tid+i+j+2),"r"(tid+i+j+3) : "memory");
#elif OP == 5
            asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                :: "l"(addr_b), "r"(tid+i+j),"r"(tid+i+j+1),"r"(tid+i+j+2),"r"(tid+i+j+3),
                                "r"(tid+i+j+4),"r"(tid+i+j+5),"r"(tid+i+j+6),"r"(tid+i+j+7) : "memory");
#elif OP == 6
            unsigned x0,x1,x2,x3;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr_a));
            asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                :: "l"(addr_b), "r"(x0),"r"(x1),"r"(x2),"r"(x3) : "memory");
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
#endif
        }
    }
#if OP < 4 || OP == 6
    C[tid] = acc0 ^ acc1 ^ acc2 ^ acc3 ^ acc4 ^ acc5 ^ acc6 ^ acc7;
#endif
}
