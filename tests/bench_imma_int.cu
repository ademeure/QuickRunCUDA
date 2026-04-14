// Integer tensor core: INT8 / INT4 / INT1 IMMA variants.

#ifndef OP
#define OP 0
#endif
#ifndef ITERS_INNER
#define ITERS_INNER 32
#endif

extern "C" __global__ __launch_bounds__(128, 2)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int a0 = threadIdx.x | 0x01010101u;
    unsigned int a1 = 0x02020202u;
    unsigned int a2 = 0x03030303u;
    unsigned int a3 = 0x04040404u;
    unsigned int b0 = 0x05050505u;
    unsigned int b1 = 0x06060606u;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ITERS_INNER; j++) {
#if OP == 0  // INT8 × INT8 m16n8k32 → S32
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 1  // U8 × S8 mixed
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 2  // INT4 × INT4 m16n8k64 → S32
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 3  // INT1 × INT1 (binary) m16n8k256
            asm volatile(
                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#endif
        }
    }
    if (c0 == 0x42424242) ((int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = c0+c1+c2+c3;
}
