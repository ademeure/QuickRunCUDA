// Blackwell FP8 / FP6 / FP4 MMA (QMMA / OMMA) via proper .kind::f8f6f4 syntax.

#ifndef OP
#define OP 0
#endif
#ifndef ITERS_INNER
#define ITERS_INNER 32
#endif

extern "C" __global__ __launch_bounds__(128, 2)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // 4 regs of A (4×8 bytes = 32 FP8 values), 2 regs B (16 FP8), 4 f32 C
    unsigned int a0 = threadIdx.x | 0x11111111u;
    unsigned int a1 = threadIdx.x | 0x22222222u;
    unsigned int a2 = threadIdx.x | 0x33333333u;
    unsigned int a3 = threadIdx.x | 0x44444444u;
    unsigned int b0 = threadIdx.x | 0x55555555u;
    unsigned int b1 = threadIdx.x | 0x66666666u;
    float c0 = 1.0f, c1 = 1.0f, c2 = 1.0f, c3 = 1.0f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ITERS_INNER; j++) {
#if OP == 0  // m16n8k32 FP8 e4m3 × e4m3 → FP32
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 1  // m16n8k32 FP8 e5m2 × e4m3 → FP32
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 2  // m16n8k64 FP4 e2m1 × e2m1 → FP32
            asm volatile(
                "mma.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 3  // m16n8k32 FP6 e2m3 × e2m3 → FP32
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m3.e2m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 4  // BF16 m16n8k16 reference (known-good)
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#endif
        }
    }
    if (c0 == 1.23456f) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = c0+c1+c2+c3;
}
