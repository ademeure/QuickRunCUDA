// Sweep: vary spacing between S2URs to characterize uniform-pipe throughput in presence
// of FFMAs. Test how many FFMAs are needed between S2URs to fully hide them.
//
// Parameter N_FMA_BETWEEN: number of parallel-chain FMAs between each S2UR.
// Total S2URs: N_S2UR (fixed). So total FMAs = N_S2UR * N_FMA_BETWEEN.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef N_S2UR
#define N_S2UR 8
#endif
#ifndef N_FMA_BETWEEN
#define N_FMA_BETWEEN 8
#endif
#ifndef ITERS
#define ITERS 4096
#endif
#ifndef CLOCK_KIND
#define CLOCK_KIND 0   /* 0 = S2UR (u32), 1 = CS2R.64 (u64) */
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    float v0 = 1.0001f + 1e-30f*(float)(tid+1);
    float v1 = 1.0001f + 1e-30f*(float)(tid+2);
    float v2 = 1.0001f + 1e-30f*(float)(tid+3);
    float v3 = 1.0001f + 1e-30f*(float)(tid+4);
    float v4 = 1.0001f + 1e-30f*(float)(tid+5);
    float v5 = 1.0001f + 1e-30f*(float)(tid+6);
    float v6 = 1.0001f + 1e-30f*(float)(tid+7);
    float v7 = 1.0001f + 1e-30f*(float)(tid+8);
    const float y = 0.9999f;
    unsigned long long acc = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int s = 0; s < N_S2UR; s++) {
            #pragma unroll
            for (int k = 0; k < N_FMA_BETWEEN; k++) {
                // cycle through the 8 chains
                switch (k & 7) {
                    case 0: v0=v0*y+v0; break;
                    case 1: v1=v1*y+v1; break;
                    case 2: v2=v2*y+v2; break;
                    case 3: v3=v3*y+v3; break;
                    case 4: v4=v4*y+v4; break;
                    case 5: v5=v5*y+v5; break;
                    case 6: v6=v6*y+v6; break;
                    case 7: v7=v7*y+v7; break;
                }
            }
#if CLOCK_KIND == 0
            unsigned c;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
            acc ^= (unsigned long long)c;
#else
            unsigned long long c;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
            acc += c;
#endif
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sum = v0+v1+v2+v3+v4+v5+v6+v7;
    if (__float_as_int(sum) == seed) C[tid] = sum;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned long long*)C)[1] = acc;
    }
}
