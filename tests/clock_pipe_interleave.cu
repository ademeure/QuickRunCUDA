// Test: does inter-leaving S2UR inside FFMA chain allow pipe overlap within a single warp?
//
// The compiler naturally places S2UR at the end of the ASM-opaque block. In single-warp
// mode, we see them serialized. Let's interleave clock reads between mini-FFMA blocks
// and see if that lets the uniform pipe run in parallel with ALU pipe within a warp.
//
// OP=0: 8 chains of 8 FMAs back-to-back (no clock, baseline)  — 76 cy
// OP=1: 8 chains x 8 FMAs, 1 S2UR at end                        — 120 cy (seen)
// OP=2: 4+4 FMAs with a S2UR between                             — does interleave help?
// OP=3: 1 S2UR per 8 FMAs block, so 8 S2URs total, heavily interleaved
// OP=4: S2UR directly after each independent FFMA (64 S2URs — extreme case)
// OP=5: interleaved CS2R.64 at same position as OP=3's S2UR (baseline for ALU)

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
#if OP == 0
        // baseline 64 FMAs, no clock
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
#elif OP == 1
        // 64 FMAs + 1 S2UR after
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        acc ^= c;
#elif OP == 2
        // 32 FMAs, S2UR, 32 FMAs
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        acc ^= c;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
#elif OP == 3
        // 8 S2URs interleaved — 8 FMAs, 1 S2UR, 8 FMAs, 1 S2UR, …
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
            unsigned c;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
            acc ^= c;
        }
#elif OP == 4
        // 64 S2URs — one per FMA per chain (extreme)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v0=v0*y+v0;
            unsigned c0; asm volatile("mov.u32 %0, %%clock;" : "=r"(c0)); acc ^= c0;
            v1=v1*y+v1;
            unsigned c1; asm volatile("mov.u32 %0, %%clock;" : "=r"(c1)); acc ^= c1;
            v2=v2*y+v2;
            unsigned c2; asm volatile("mov.u32 %0, %%clock;" : "=r"(c2)); acc ^= c2;
            v3=v3*y+v3;
            unsigned c3; asm volatile("mov.u32 %0, %%clock;" : "=r"(c3)); acc ^= c3;
            v4=v4*y+v4;
            unsigned c4; asm volatile("mov.u32 %0, %%clock;" : "=r"(c4)); acc ^= c4;
            v5=v5*y+v5;
            unsigned c5; asm volatile("mov.u32 %0, %%clock;" : "=r"(c5)); acc ^= c5;
            v6=v6*y+v6;
            unsigned c6; asm volatile("mov.u32 %0, %%clock;" : "=r"(c6)); acc ^= c6;
            v7=v7*y+v7;
            unsigned c7; asm volatile("mov.u32 %0, %%clock;" : "=r"(c7)); acc ^= c7;
        }
#elif OP == 5
        // 8 CS2R.64 interleaved — counterpart to OP=3 but ALU pipe
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
            unsigned long long c;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
            acc += c;
        }
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
