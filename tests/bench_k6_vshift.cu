// K6: vshl/vshr (vector shift) PTX → SASS
// PTX vshl/vshr.s32.s32.s32 perform shifts on packed bytes/halves
// Test: do these compile to native SASS or are they emulated?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)(threadIdx.x ^ u2);
    unsigned int x = 0xDEADBEEFu ^ (unsigned)u2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // PTX vshl.u32 with clamp
            asm volatile("vshl.u32.u32.u32.clamp %0, %0, %1;" : "+r"(v) : "r"(x));
#elif MODE == 1
            // PTX vshr.u32 with clamp
            asm volatile("vshr.u32.u32.u32.clamp %0, %0, %1;" : "+r"(v) : "r"(x));
#elif MODE == 2
            // Plain shl.b32 (regular scalar shift)
            asm volatile("shl.b32 %0, %0, %1;" : "+r"(v) : "r"(x));
#elif MODE == 3
            // Plain shr.u32
            asm volatile("shr.u32 %0, %0, %1;" : "+r"(v) : "r"(x));
#elif MODE == 4
            // shf.l.wrap.b32 (funnel shift)
            asm volatile("shf.l.wrap.b32 %0, %0, %1, 4;" : "+r"(v) : "r"(x));
#elif MODE == 5
            // PTX vadd4.u32 (4-byte vector add)
            asm volatile("vadd4.u32.u32.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(x), "r"(0u));
#elif MODE == 6
            // PTX vmin4 (4-byte vector min)
            asm volatile("vmin4.u32.u32.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(x), "r"(0u));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/op=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS/16.0);
    }
}
