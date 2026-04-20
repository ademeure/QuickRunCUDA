// st family SASS encoding comparison
// Mode 0: st.global (default)
// Mode 1: st.shared
// Mode 2: st.local
// Mode 3: st.global.cs (cache-streaming)
// Mode 4: st.shared.relaxed.cta
// Mode 5: st.shared.b128 (vectorized)

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int* p = (int*)A;
    __shared__ int smem[1024];
    int local[64];
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    int val = idx + (int)u2;

#if MODE == 0
    asm volatile("st.global.u32 [%0], %1;" :: "l"(p + idx), "r"(val));
#elif MODE == 1
    asm volatile("st.shared.u32 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(smem + (idx & 1023))), "r"(val));
#elif MODE == 2
    asm volatile("st.local.u32 [%0], %1;" :: "l"(local + (idx & 63)), "r"(val));
#elif MODE == 3
    asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(p + idx), "r"(val));
#elif MODE == 4
    asm volatile("st.shared.relaxed.cta.u32 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(smem + (idx & 1023))), "r"(val));
#elif MODE == 5
    int4 vec = make_int4(val, val+1, val+2, val+3);
    asm volatile("st.shared.b128 [%0], {%1,%2,%3,%4};"
                 :: "r"((unsigned)__cvta_generic_to_shared(smem + (idx & 1020))),
                    "r"(vec.x), "r"(vec.y), "r"(vec.z), "r"(vec.w));
#endif

    if (threadIdx.x == 0) C[0] = (float)smem[0] + (float)local[0];
}
