// Multi-warp test: do S2UR reads from one warp steal cycles from FFMA in parallel warps?
// Each warp runs an FFMA-heavy loop and does its own clock reads; we measure total cy
// by lane-0 of block 0.
//
// Because all 4 warp schedulers are busy with FFMAs, if S2UR truly runs on a separate
// pipe that's otherwise idle, S2UR should be almost free.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128   // 4 warps
#endif
#ifndef OP
#define OP 0
#endif
#ifndef ITERS
#define ITERS 4096
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
#if OP != 100 /* all ops do FFMA */
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
#endif
#if OP == 0
        // no clock
#elif OP == 1
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        acc ^= c;
#elif OP == 2
        unsigned long long c;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        acc += c;
#endif
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sum = v0+v1+v2+v3+v4+v5+v6+v7;
    if (__float_as_int(sum) == seed) C[tid] = sum;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned long long*)C)[1] = acc;
    }
}
