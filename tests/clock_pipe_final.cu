// Final head-to-head: place a SINGLE clock read in the middle of the FMA stream.
// Compare S2UR vs CS2R.32 (both routed via consumer pattern).
//
// OP=0: 64 FMAs, no clock (baseline)
// OP=1: 32 FMA | S2UR (mov.u32, acc^=c) | 32 FMA  — uniform pipe
// OP=2: 32 FMA | CS2R.64 (mov.u64, acc+=c) | 32 FMA — ALU pipe, full 64
// OP=3: 32 FMA | CS2R.32 (mov.u32 fed to FFMA) | 32 FMA — ALU pipe, 32-bit

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
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
        // first 32 FMAs
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
#if OP == 1
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        acc ^= c;
#elif OP == 2
        unsigned long long c;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        acc += c;
#elif OP == 3
        // Force CS2R.32 via ALU consumer
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        if (__float_as_int(v0) == seed) v0 += __int_as_float(c) * 1e-30f;
        acc ^= c;
#endif
        // second 32 FMAs
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
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
