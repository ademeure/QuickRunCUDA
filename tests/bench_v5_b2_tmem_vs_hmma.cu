// V5 B2: TMEM access (tcgen05.ld) vs HMMA pipe competition
// Test: does tcgen05.ld occupy the tensor pipe or is it separate?
// MODE 0: HMMA only (tensor pipe baseline)
// MODE 1: tcgen05.ld only (TMEM read pipe)
// MODE 2: HMMA + tcgen05.ld interleaved (overlap test)
//
// Since we have the cycles measure + ncu sm__pipe_tensor metric,
// we can see if MODE 2 = MODE 0 + MODE 1 (serial) or max() (parallel).
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ unsigned int tmem_addr;
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
        :: "r"((unsigned int)__cvta_generic_to_shared(&tmem_addr))
    );
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    __syncwarp();

    // Initialize HMMA accumulator
    unsigned int a0=0x3F803F80, a1=0x3F803F80, a2=0x3F803F80, a3=0x3F803F80;
    unsigned int b0=0x3F803F80, b1=0x3F803F80;
    float c0=0.0f, c1=0.0f, c2=0.0f, c3=0.0f;

    // Pre-fill TMEM with some data using tcgen05.st (write 4 b32 from regs)
    // tcgen05.st.sync.aligned.32x32b.x4.b32 [tmem_addr], {r0, r1, r2, r3}
    unsigned int t0r=0xCAFEBABE, t1r=0xDEADBEEF, t2r=0x12345678, t3r=0xABCDEF01;
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr), "r"(t0r), "r"(t1r), "r"(t2r), "r"(t3r)
    );
    asm volatile("tcgen05.wait::st.sync.aligned;");
    __syncwarp();

    // Storage for tcgen05.ld results
    unsigned int ld0=0, ld1=0, ld2=0, ld3=0;

    unsigned long long ts0, ts1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(ts0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0 || MODE == 2
        // HMMA chain (4 ops for ILP)
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
#endif
#if MODE == 1 || MODE == 2
        // tcgen05.ld chain — accumulate so CSE can't remove any
        unsigned int t0,t1,t2,t3;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(t0), "=r"(t1), "=r"(t2), "=r"(t3)
                : "r"(tmem_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            ld0 ^= t0; ld1 ^= t1; ld2 ^= t2; ld3 ^= t3;
        }
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(ts1));

    // Anti-DCE: independently force use of c[] (HMMA) and ld[] (TMEM)
#if MODE == 0 || MODE == 2
    if (c0 == 1.234567e-30f) C[blockIdx.x] = c0 + c1 + c2 + c3;
#endif
#if MODE == 1 || MODE == 2
    unsigned int sum_ld = ld0 + ld1 + ld2 + ld3;
    if (sum_ld == 0xCAFEBABE) ((unsigned int*)C)[blockIdx.x + 1] = sum_ld;
#endif

    // Dealloc
    __syncwarp();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(tmem_addr));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d total_cy=%llu cy/iter=%.3f\n",
               MODE, ts1-ts0, (double)(ts1-ts0)/(double)ITERS);
    }
}
