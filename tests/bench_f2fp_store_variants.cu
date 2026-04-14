// F2FP (unpack) + 1 store per iter, trying different store variants.
// STORE_TYPE: 0=st.global.f32        (default)
//             1=st.global.cs.f32     (streaming, bypass L1)
//             2=st.global.cg.f32     (cache-global, skip L1)
//             3=st.global.wt.f32     (write-through)
//             4=st.shared.f32        (shared memory)
//             5=LDG                  (load from global, for comparison)
//             6=red.shared.or.b32    (shared atomic)
//             7=st.global.v4.f32     (wider store)
//             8=st.local.f32         (local memory store — uses stack)

#ifndef STORE_TYPE
#define STORE_TYPE 0
#endif
#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef N_STORE
#define N_STORE 0
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(const float* __restrict__ A, float* B,
            float* __restrict__ C, int ITERS, int seed, int unused_2) {
    unsigned short h[N_CHAINS];
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) h[k] = 0x3C01 ^ (threadIdx.x + k);

#if N_STORE > 0 && STORE_TYPE != 4 && STORE_TYPE != 6
    float* my_C = C + (threadIdx.x & 0xFF) + 1024 + blockIdx.x * 1024;
    float val = (float)threadIdx.x * 0.001f;
#endif
#if STORE_TYPE == 4 || STORE_TYPE == 6
    __shared__ float smem[BLOCK_SIZE + 32];
    float val = (float)threadIdx.x * 0.001f;
#endif
#if STORE_TYPE == 5
    const float* my_A = A + (threadIdx.x & 0xFF) + 1024;
    float ldval;
#endif
#if STORE_TYPE == 7
    float4 v4val = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    float4* my_C4 = (float4*)(C + blockIdx.x * 1024 + 2048);
#endif
#if STORE_TYPE == 8
    float loc[4] = {1.0f, 2.0f, 3.0f, 4.0f};
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // F2FP unpack (baseline 64/SM/clk)
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(p[k]) : "h"(h[k]));
                h[k] = (unsigned short)p[k];
            }
#if N_STORE > 0
            #pragma unroll
            for (int m = 0; m < N_STORE; m++) {
  #if STORE_TYPE == 0
                asm volatile("st.global.f32 [%0], %1;" :: "l"(my_C + m), "f"(val));
  #elif STORE_TYPE == 1
                asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(my_C + m), "f"(val));
  #elif STORE_TYPE == 2
                asm volatile("st.global.cg.f32 [%0], %1;" :: "l"(my_C + m), "f"(val));
  #elif STORE_TYPE == 3
                asm volatile("st.global.wt.f32 [%0], %1;" :: "l"(my_C + m), "f"(val));
  #elif STORE_TYPE == 4
                asm volatile("st.shared.f32 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(&smem[m])), "f"(val));
  #elif STORE_TYPE == 5
                asm volatile("ld.global.f32 %0, [%1];" : "=f"(ldval) : "l"(my_A + m));
                val ^= (int)ldval;
  #elif STORE_TYPE == 6
                asm volatile("atom.shared.or.b32 %0, [%1], %2;" : "=r"(*(unsigned*)&val)
                             : "r"((unsigned)__cvta_generic_to_shared(&smem[m])), "r"(0x1));
  #elif STORE_TYPE == 7
                asm volatile("st.global.v4.f32 [%0], {%1,%2,%3,%4};"
                             :: "l"(my_C4 + m), "f"(v4val.x), "f"(v4val.y), "f"(v4val.z), "f"(v4val.w));
  #elif STORE_TYPE == 8
                loc[m & 3] = val + (float)m;
  #endif
            }
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= (unsigned int)h[k];
#if STORE_TYPE == 5 || STORE_TYPE == 4 || STORE_TYPE == 6
    acc ^= __float_as_int(val);
#endif
#if STORE_TYPE == 8
    acc ^= __float_as_int(loc[0] + loc[1] + loc[2] + loc[3]);
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
