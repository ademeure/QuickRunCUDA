// K5: st.shared vs st.global SASS encoding family
// Compare PTX store variants → SASS opcodes; latency
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) unsigned int smem[1024];
    unsigned int smem_addr = __cvta_generic_to_shared(smem);
    unsigned int v = (unsigned)(threadIdx.x ^ u2);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned int off = (i & 31) * 4;
        v += i;
#if MODE == 0
        // st.shared.u32
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_addr + off), "r"(v));
#elif MODE == 1
        // st.global.u32 (default = strong.gpu)
        asm volatile("st.global.u32 [%0], %1;" :: "l"(C + off/4), "r"(v));
#elif MODE == 2
        // st.global.cg (cache global, L1 bypass)
        asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(C + off/4), "r"(v));
#elif MODE == 3
        // st.global.cs (streaming)
        asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(C + off/4), "r"(v));
#elif MODE == 4
        // st.global.wt (write-through)
        asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(C + off/4), "r"(v));
#elif MODE == 5
        // st.shared with cta scope (default)
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_addr + off), "r"(v));
#elif MODE == 6
        // STG via vec4 store (uint4 = 16 B)
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(smem_addr + off), "r"(v), "r"(v), "r"(v), "r"(v));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[1024] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/store=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
