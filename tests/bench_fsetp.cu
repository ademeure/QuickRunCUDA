// FSETP variants
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;
    float fb = 1.000001f + (float)u2 * 1e-9f;
    unsigned int sel = 0;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // setp.eq (ordered equal — NaN gives false)
        asm("{ .reg .pred p; setp.eq.f32 p, %1, %2; selp.b32 %0, 1, 0, p; }"
            : "=r"(sel) : "f"(fv), "f"(fb));
        fv += (float)sel;
#elif MODE == 1
        // setp.equ (unordered equal — NaN gives true)
        asm("{ .reg .pred p; setp.equ.f32 p, %1, %2; selp.b32 %0, 1, 0, p; }"
            : "=r"(sel) : "f"(fv), "f"(fb));
        fv += (float)sel;
#elif MODE == 2
        // setp.gt (ordered greater)
        asm("{ .reg .pred p; setp.gt.f32 p, %1, %2; selp.b32 %0, 1, 0, p; }"
            : "=r"(sel) : "f"(fv), "f"(fb));
        fv += (float)sel;
#elif MODE == 3
        // setp.gtu (unordered greater)
        asm("{ .reg .pred p; setp.gtu.f32 p, %1, %2; selp.b32 %0, 1, 0, p; }"
            : "=r"(sel) : "f"(fv), "f"(fb));
        fv += (float)sel;
#elif MODE == 4
        // setp.num (test if both ordered)
        asm("{ .reg .pred p; setp.num.f32 p, %1, %2; selp.b32 %0, 1, 0, p; }"
            : "=r"(sel) : "f"(fv), "f"(fb));
        fv += (float)sel;
#elif MODE == 5
        // setp.nan (test if either is NaN)
        asm("{ .reg .pred p; setp.nan.f32 p, %1, %2; selp.b32 %0, 1, 0, p; }"
            : "=r"(sel) : "f"(fv), "f"(fb));
        fv += (float)sel;
#endif
    }

    if ((int)fv == seed && sel == (unsigned)seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
