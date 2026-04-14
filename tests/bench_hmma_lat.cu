// HMMA (tensor core) latency — inline mma.sync with clock64 chain.

#ifndef N_OPS
#define N_OPS 64
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 256
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // 32-thread warp participates in mma.sync
    // Per-lane: 4 regs of A fragment, 2 regs of B, 4 f32 C
    unsigned int a0 = threadIdx.x * 0x01010101, a1 = 0x02020202, a2 = 0x03030303, a3 = 0x04040404;
    unsigned int b0 = threadIdx.x * 0x05050505, b1 = 0x06060606;
    float c0 = 1.0f, c1 = 2.0f, c2 = 3.0f, c3 = 4.0f;
    unsigned long long total_dt = 0;

    // Only thread 0 measures, but all threads participate in MMAs
    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0 = 0, t1 = 0;
        if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0  // FP16 m16n8k16 → FP32
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 1  // BF16 m16n8k16 → FP32
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
#elif OP == 2  // TF32 m16n8k8 → FP32
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1), "r"(b0));
#elif OP == 3  // FP16 m16n8k8 (smaller K)
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1), "r"(b0));
#elif OP == 4  // INT8 m16n8k32 → S32
            int i0=1,i1=2,i2=3,i3=4;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(i0),"+r"(i1),"+r"(i2),"+r"(i3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
            c0 = (float)i0;
#endif
        }
        if (threadIdx.x == 0) {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
            total_dt += (t1 - t0);
        }
    }
    if (threadIdx.x == 0) {
        if (c0 == 1.23456f) ((float*)C)[0] = c0+c1+c2+c3;
        ((unsigned long long*)C)[1] = total_dt;
    }
}
