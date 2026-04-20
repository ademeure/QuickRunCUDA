// cvt chains: when does PTX cvt sequence fuse to one SASS inst?

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    int x = idx + (int)u2;
    float fy;

#if MODE == 0
    // f32 -> s32 (single cvt)
    fy = (float)idx;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(x) : "f"(fy));
#elif MODE == 1
    // f32 -> f16 -> f32 (round-trip via f16)
    asm("{ .reg .f16 t; cvt.rn.f16.f32 t, %1; cvt.f32.f16 %0, t; }" : "=f"(fy) : "f"((float)idx));
    x = (int)fy;
#elif MODE == 2
    // f32 -> f16 -> bf16 (cross-format chain)
    asm("{ .reg .f16 a; .reg .bf16 b; cvt.rn.f16.f32 a, %1; cvt.rn.bf16.f32 b, %1; "
        "  cvt.f32.bf16 %0, b; }" : "=f"(fy) : "f"((float)idx));
    x = (int)fy;
#elif MODE == 3
    // s32 -> f32 -> s32 (round-trip via f32)
    fy = (float)idx;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(x) : "f"(fy));
#elif MODE == 4
    // s32 -> u8 -> s32 (saturating then back)
    asm("{ .reg .s32 t; cvt.sat.u8.s32 t, %1; cvt.s32.u32 %0, t; }" : "=r"(x) : "r"(idx));
#endif

    C[idx] = (float)x;
}
