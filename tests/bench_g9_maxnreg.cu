// G9: __launch_bounds__ maxnreg vs default register count
// Try kernel that wants many registers; force lower count via maxnreg
// Measure runtime + check SASS register count
#ifndef MAXREG
#define MAXREG 0
#endif

// minBlocks controls implicit reg budget: regs_per_thread = 65536 / minBlocks / 256
// minBlocks=8 → 32 regs; 4 → 64; 2 → 128; 1 → 256 (capped at 255)
#if MAXREG == 0
#define LB __launch_bounds__(256, 1)  // no constraint baseline
#elif MAXREG == 32
#define LB __launch_bounds__(256, 8)
#elif MAXREG == 64
#define LB __launch_bounds__(256, 4)
#elif MAXREG == 128
#define LB __launch_bounds__(256, 2)
#elif MAXREG == 255
#define LB __launch_bounds__(256, 1)
#endif

extern "C" __global__ LB
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // 32 indep chains + 32 distinct ya/za to force high register pressure
    float a[32], ya[32], za[32];
    for (int k = 0; k < 32; k++) {
        a[k] = (float)(threadIdx.x ^ u2) * 0.001f * (k+1);
        ya[k] = (float)(threadIdx.x ^ (u2+k)) * 0.002f + 1.0f;
        za[k] = 0.5f + (float)k * 0.001f;
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            a[k] = a[k] * ya[k] + za[k];
        }
    }

    float sink = 0;
    for (int k = 0; k < 32; k++) sink += a[k];
    if ((int)sink == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = sink;
}
