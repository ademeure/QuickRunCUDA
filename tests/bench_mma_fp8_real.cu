// FP8 mma.sync via kind::f8f6f4 — defeat constant fold by varying operands per-iter
// Discovery: kind::f8f6f4 on sm_103a compiles to F2FP unpack + HMMA.16816 (FP16 path)
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(128, 2)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int a0 = (threadIdx.x ^ u2) | 0x11111111u;
    unsigned int a1 = (threadIdx.x ^ u2) | 0x22222222u;
    unsigned int a2 = (threadIdx.x ^ u2) | 0x33333333u;
    unsigned int a3 = (threadIdx.x ^ u2) | 0x44444444u;
    unsigned int b0 = (threadIdx.x ^ u2) | 0x55555555u;
    unsigned int b1 = (threadIdx.x ^ u2) | 0x66666666u;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Rotate operands each iter to defeat constant-fold
        a0 ^= i; a1 ^= i; b0 ^= i;
#if OP == 0  // FP8 e4m3 m16n8k32 (kind::f8f6f4)
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 1  // BF16 m16n8k16 reference
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 2  // BF16 m16n8k16 + 2 mma per iter (k16 doubled = K=32 like FP8)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)c0 == seed) C[blockIdx.x] = c0+c1+c2+c3;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("OP=%d clk=%llu cy/iter=%.2f\n", OP, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
