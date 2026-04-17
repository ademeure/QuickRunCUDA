// Constant memory audit with multiple access patterns:
// OP=0: Broadcast (every thread same address) — 4 B/LDC per warp
// OP=1: Per-warp distinct (threads in warp see same, warps see different)
// OP=2: Per-thread distinct via threadIdx — bank conflicts?
// OP=3: LDC.v4 (16 B per LDC) via inline asm
// OP=4: LDC.v4 broadcast pattern

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef OP
#define OP 0
#endif

__device__ __constant__ uint4 cmem4[4096];      // 64 KB

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int ITERS, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = tid & 31;
    unsigned acc0=0, acc1=0, acc2=0, acc3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0
            // Broadcast: every thread same idx
            unsigned idx = (i + j) & 16383;
            acc0 += ((unsigned*)cmem4)[idx];
#elif OP == 1
            // Per-warp distinct: idx depends on warpid
            unsigned idx = ((i + j) * 32 + (tid >> 5)) & 16383;
            acc0 += ((unsigned*)cmem4)[idx];
#elif OP == 2
            // Per-thread distinct: idx depends on tid (will cause cmem serialization!)
            unsigned idx = ((i + j) + tid) & 16383;
            acc0 += ((unsigned*)cmem4)[idx];
#elif OP == 3
            // C++-level uint4 read from cmem — should emit LDC.128
            unsigned idx = (i + j) & 4095;
            uint4 v = cmem4[idx];
            acc0 += v.x; acc1 += v.y; acc2 += v.z; acc3 += v.w;
#elif OP == 4
            // 4 separate LDC.32 with sequential reads — should be coalesced into LDC.128
            unsigned idx = (i + j) & 4095;
            unsigned* p = (unsigned*)&cmem4[idx];
            acc0 += p[0]; acc1 += p[1]; acc2 += p[2]; acc3 += p[3];
#elif OP == 5
            // LDC.64 via uint2
            unsigned idx = (i + j) & 8191;
            uint2 v = ((uint2*)cmem4)[idx];
            acc0 += v.x; acc1 += v.y;
#elif OP == 6
            // Try forcing LDC.128 via inline PTX
            unsigned idx = (i + j) & 4095;
            unsigned x0,x1,x2,x3;
            asm volatile("ld.const.v4.u32 {%0,%1,%2,%3}, [cmem4 + %4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "n"(0));
            // The address [cmem4 + N] needs to be a constant for `ld.const`...
            // try with index variable
            (void)idx;
            acc0 += x0; acc1 += x1; acc2 += x2; acc3 += x3;
#elif OP == 7
            // Try LDC.128 by reading 4× LDC.32 sequentially (compiler may merge)
            unsigned idx = (i + j) & 4095;
            unsigned* p = (unsigned*)&cmem4[idx];
            acc0 += __ldg((unsigned*)&cmem4[idx]);  // wrong — __ldg is for global
            (void)p;
#endif
        }
    }
    C[tid] = acc0 ^ acc1 ^ acc2 ^ acc3;
}
