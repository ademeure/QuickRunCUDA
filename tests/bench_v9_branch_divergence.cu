// V9: Branch divergence cost
// Compare warp with 32 threads same path vs various divergence patterns.
#ifndef PATTERN
#define PATTERN 0   // 0=all same, 1=2-way split, 2=4-way, 3=8-way, 4=32-way
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    int lane = threadIdx.x & 31;
    float v = (float)(lane + 1) * 0.001f;
    float b = (float)(lane + 2) * 0.001f;

    // Determine divergence group per lane
#if PATTERN == 0
    int group = 0;  // All same path
#elif PATTERN == 1
    int group = lane & 1;  // 2-way split
#elif PATTERN == 2
    int group = lane & 3;  // 4-way
#elif PATTERN == 3
    int group = lane & 7;  // 8-way
#elif PATTERN == 4
    int group = lane;      // 32-way (max divergence)
#endif

    unsigned long long t0, t1;
    if (lane == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    __syncwarp();

    // Each group does different FFMA path
    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
        // 32-way switch — each group takes different branch
        // SASS should emit SSY/SYNC or predicated reconvergence
        switch (group) {
            case 0:  v = v * b + 0.1f; break;
            case 1:  v = v * b + 0.2f; break;
            case 2:  v = v * b + 0.3f; break;
            case 3:  v = v * b + 0.4f; break;
            case 4:  v = v * b + 0.5f; break;
            case 5:  v = v * b + 0.6f; break;
            case 6:  v = v * b + 0.7f; break;
            case 7:  v = v * b + 0.8f; break;
            case 8:  v = v * b + 0.9f; break;
            case 9:  v = v * b + 1.0f; break;
            case 10: v = v * b + 1.1f; break;
            case 11: v = v * b + 1.2f; break;
            case 12: v = v * b + 1.3f; break;
            case 13: v = v * b + 1.4f; break;
            case 14: v = v * b + 1.5f; break;
            case 15: v = v * b + 1.6f; break;
            default: v = v * b + 2.0f; break;
        }
    }

    __syncwarp();
    if (lane == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
    }
    if (v == 1.234567e-30f) C[lane + 4] = v;
}
