// LDS (shared-mem load) latency + register operand port contention test.

#ifndef N_OPS
#define N_OPS 128
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 256
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // Pre-populate shared memory
    for (int i = 0; i < 128; i++) smem[i] = i * 0x11 + 1;
    __syncthreads();

    unsigned int addr = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned int x = 0;
    float f = 1.0f;
    unsigned long long total_dt = 0;

    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0  // LDS chain: each load depends on previous (addr = prev_value)
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(addr + ((x & 0x7F) << 2)));
#elif OP == 1  // LDS chain: but fixed address (no dep, just back-to-back)
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(addr));
#elif OP == 2  // LDS.128 chain
            unsigned int a,b,c,d;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(addr + ((x & 0xF) << 2)));
            x = a^b^c^d;
#elif OP == 3  // LDS + ADD (common pattern)
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(addr + ((x & 0x7F) << 2)));
            x = x + 1;
#elif OP == 4  // FFMA with 4 distinct input regs (no port contention)
            float f0 = f + 0.01f, f1 = f + 0.02f, f2 = f + 0.03f;
            asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(f) : "f"(f0), "f"(f1), "f"(f2));
#elif OP == 5  // FFMA with same reg in all 3 inputs (port contention?)
            asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(f));
#elif OP == 6  // FFMA with same reg in 2 inputs
            asm volatile("fma.rn.f32 %0, %0, %0, %1;" : "+f"(f) : "f"(1.1f));
#elif OP == 7  // LDC (constant bank load) — likely free
            asm volatile("ld.const.u32 %0, [%1];" : "=r"(x) : "l"(4));
#endif
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }
    if ((int)x == seed || __float_as_int(f) == seed) ((unsigned int*)C)[0] = x + __float_as_int(f);
    ((unsigned long long*)C)[1] = total_dt;
}
