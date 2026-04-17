// Force S2UR vs CS2R path for the SAME scenario (clock read at end of FFMA chain).
// Compare timing directly.
//
// Trick to force CS2R.32: cvt.u64.u32 dummy chain or direct assignment into ALU.
// Trick to force S2UR:    use %x or %xh in ptx with uniform register or "=h" modifiers?
//                          Easiest is: bind to a uniform predicate via asm "+l" then
//                          strip — but compiler has final say. Alternative: use ptx
//                          "volatile" with .u32 (already tried) vs use .u32 with
//                          sethi.b32 (not meaningful on volta+).
//
// Approach: compare the 3 versions side-by-side with explicit PTX.

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
        // 64 FMAs
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v0=v0*y+v0; v1=v1*y+v1; v2=v2*y+v2; v3=v3*y+v3;
            v4=v4*y+v4; v5=v5*y+v5; v6=v6*y+v6; v7=v7*y+v7;
        }
#if OP == 0
        // baseline: no clock read
#elif OP == 1
        // u32 clock (compiler picks S2UR since result consumed by u64 acc)
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        acc ^= c;
#elif OP == 2
        // u64 clock (always CS2R)
        unsigned long long c;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        acc += c;
#elif OP == 3
        // Force CS2R.32: feed clock into a float via __int_as_float -> FFMA
        // The consumer being ALU (FFMA) should push compiler to CS2R.32
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        // Mix into v0 — but only under impossible condition to avoid affecting FMA throughput
        if (__float_as_int(v0) == seed) v0 += __int_as_float(c) * 1e-30f;
        acc ^= c;
#elif OP == 4
        // Force S2UR: explicit ptx uniform intent via "bar.warp" fence or similar
        // Simplest: multiple back-to-back u32 reads sharing uniform context
        unsigned c1, c2;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c1));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c2));
        acc ^= c1 ^ c2;
#elif OP == 5
        // u64 clock consumed only by low 32 bits (does compiler switch to CS2R.32?)
        unsigned long long c;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        acc ^= (unsigned)c;
#elif OP == 6
        // Explicit uniform reg binding via .reg .pred trick — limited PTX
        // "mov.u32 %0, %%clock" with result fed to uniform ops
        unsigned c;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
        // Force uniform use by doing a uniform shift/rotate on a "broadcast" value
        // In CUDA there's no portable way; try bit-reversal which generates SASS BREV
        unsigned brev;
        asm volatile("brev.b32 %0, %1;" : "=r"(brev) : "r"(c));
        acc ^= brev;
#elif OP == 7
        // Read u64 and force full-width use via xor of hi and lo
        unsigned long long c;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(c));
        acc ^= (unsigned)c ^ (unsigned)(c >> 32);
#elif OP == 8
        // Alternate lo and hi reads (no shift)
        unsigned lo, hi;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(lo));
        asm volatile("mov.u32 %0, %%clock_hi;" : "=r"(hi));
        acc ^= (unsigned long long)lo ^ ((unsigned long long)hi << 32);
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
