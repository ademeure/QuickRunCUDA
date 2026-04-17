// Isolation test: measure parallel pipe overlap of clock reads vs FFMA work.
//
// Each iteration runs 8 parallel FFMA chains (8 FMAs each = 64 FMAs/iter, back-to-back,
// 8-way ILP so FMA throughput ≈ 1 cy/FMA when FMA dispatch is the bottleneck).
// Then we add 0, 1, 4, or 8 clock reads per iter and see cy/iter.
//
// Variants:
//   OP=0  : FFMA-only baseline (NO clock reads per iter — only t0/t1 brackets)
//   OP=1  : FFMA + 1 mov.u64 %clock64 (expected CS2R R, SR_CLOCKLO, full u64 use)
//   OP=2  : FFMA + 1 mov.u32 %clock (may emit S2UR or CS2R.32 depending on heuristic)
//   OP=3  : FFMA + 4 mov.u64 %clock64 (heavy ALU clock contention)
//   OP=4  : FFMA + 4 mov.u32 %clock  (try to get 4 S2UR reads)
//   OP=5  : FFMA + 1 mov.u32 %clock with UNIFORM-PIPE use (UIADD3 via ptx "uni" pred attempt)
//   OP=6  : FFMA + 8 mov.u32 %clock (heavy uniform-pipe pressure)
//   OP=7  : FFMA + 1 mov.u32 %clock, result fed back into FFMA chain (force ALU move)
//   OP=8  : FFMA + 1 mov.u64 %clock64 but only using LOW 32 bits (tests whether
//           compiler still emits CS2R with both halves)
//   OP=9  : NO FFMA at all, 8 mov.u32 %clock (isolate uniform-pipe solo)
//   OP=10 : NO FFMA at all, 8 mov.u64 %clock64 (isolate ALU CS2R solo)

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
    // 8 independent FFMA chains
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
#if OP != 9 && OP != 10
        // 64 FMAs in 8 parallel chains (8 ILP chains x 8 deep)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v0 = v0*y + v0; v1 = v1*y + v1; v2 = v2*y + v2; v3 = v3*y + v3;
            v4 = v4*y + v4; v5 = v5*y + v5; v6 = v6*y + v6; v7 = v7*y + v7;
        }
#endif
#if OP == 1
        unsigned long long c;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        acc += c;  // forces use of both halves
#elif OP == 2
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        acc ^= (unsigned long long)c;
#elif OP == 3
        unsigned long long c1, c2, c3, c4;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c1));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c2));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c3));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c4));
        acc += c1 + c2 + c3 + c4;
#elif OP == 4
        unsigned c1, c2, c3, c4;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c1));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c2));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c3));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c4));
        acc ^= c1 ^ c2 ^ c3 ^ c4;
#elif OP == 5
        // Keep c in a uniform-pipe-native accumulator: use xor into acc via uniform ops.
        // Inline asm with .pred use on uniform reg is hard; instead we mix into a second
        // u32 accumulator that's only touched by u32 ops.
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        acc ^= (unsigned long long)c;
#elif OP == 6
        unsigned c1, c2, c3, c4, c5, c6, c7, c8;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c1));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c2));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c3));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c4));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c5));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c6));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c7));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c8));
        acc ^= c1 ^ c2 ^ c3 ^ c4 ^ c5 ^ c6 ^ c7 ^ c8;
#elif OP == 7
        // Feed clock into FFMA chain (forces c to be available before next round)
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        // Mix c into v0 to create cross-pipe dep
        v0 = v0 + __int_as_float(c) * 1e-30f;
        acc ^= (unsigned long long)c;
#elif OP == 8
        // Read u64 clock but use only low 32 bits (tests if compiler can elide high half)
        unsigned long long c;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        acc ^= (unsigned)c;  // only low 32
#elif OP == 9
        // No FFMA: pure uniform pipe stress (8 S2UR per iter)
        unsigned c1, c2, c3, c4, c5, c6, c7, c8;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c1));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c2));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c3));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c4));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c5));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c6));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c7));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c8));
        acc ^= c1 ^ c2 ^ c3 ^ c4 ^ c5 ^ c6 ^ c7 ^ c8;
#elif OP == 10
        // No FFMA: pure ALU pipe stress (8 CS2R per iter)
        unsigned long long c1, c2, c3, c4, c5, c6, c7, c8;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c1));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c2));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c3));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c4));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c5));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c6));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c7));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c8));
        acc += c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8;
#endif
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Persist (prevent DCE)
    float sum = v0+v1+v2+v3+v4+v5+v6+v7;
    if (__float_as_int(sum) == seed) C[tid] = sum;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned long long*)C)[1] = acc;
    }
}
