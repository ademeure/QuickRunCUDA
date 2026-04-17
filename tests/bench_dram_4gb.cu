#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef OP
#define OP 0
#endif
#ifndef WS_GB
#define WS_GB 1
#endif

#if OP == 1 || OP == 3 || OP == 5
#define BPI 32
#else
#define BPI 16
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned long long ws_bytes = (unsigned long long)WS_GB * 1073741824ULL;
    unsigned long long ws_mask = ws_bytes - 1ULL;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned acc0=0, acc1=0, acc2=0, acc3=0, acc4=0, acc5=0, acc6=0, acc7=0;
    int ITERS = u0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned long long off = (((unsigned long long)tid * BPI + (unsigned long long)(i + j) * BPI * gridDim.x * blockDim.x) & ws_mask);
            off = off & ~(BPI - 1ULL);
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
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3; acc4^=x4; acc5^=x5; acc6^=x6; acc7^=x7;
#elif OP == 5
            asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                :: "l"(addr_b), "r"(tid+i+j),"r"(tid+i+j+1),"r"(tid+i+j+2),"r"(tid+i+j+3),
                                "r"(tid+i+j+4),"r"(tid+i+j+5),"r"(tid+i+j+6),"r"(tid+i+j+7) : "memory");
#endif
        }
    }
#if OP < 4
    C[tid] = acc0 ^ acc1 ^ acc2 ^ acc3 ^ acc4 ^ acc5 ^ acc6 ^ acc7;
#endif
}
