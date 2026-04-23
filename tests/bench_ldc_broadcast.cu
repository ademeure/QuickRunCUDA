// B9 audit: Constant memory broadcast LDC.32 throughput
// Catalog claims 17.8 TB/s effective (32-lane broadcast amplification),
// ~0.55 TB/s actual cache traffic.
//
// Two variants via -H "#define MODE N":
//   MODE=0 : warp-uniform index (i+u0)&255 -> ptxas should emit LDCU (uniform RF)
//   MODE=1 : per-lane varying index (i+tid)&255 -> ptxas should emit LDC.32 to per-thread
//   MODE=2 : per-lane varying index, broadcast across warp via tid masked off (test broadcast detection)
//
// Header injection sets BS, NC (per-iter loads), N_ITERS

#ifndef MODE
#define MODE 0
#endif
#ifndef N_ITERS
#define N_ITERS 4096
#endif
#ifndef NC
#define NC 8
#endif
#ifndef BS
#define BS 256
#endif

__constant__ unsigned int CMEM[256];

extern "C" __global__ __launch_bounds__(BS, 4)
void kernel(float* A, float* B, float* C, int u0, int u1, int u2) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sum = 0;

    #pragma unroll 1
    for (int i = 0; i < N_ITERS; i++) {
#if MODE == 0
        // Uniform-across-warp index. u0 is runtime-variable; ptxas should detect uniform → LDCU
        unsigned int base = (i + u0);
        #pragma unroll
        for (int k = 0; k < NC; k++) {
            sum ^= CMEM[(base + k) & 255];
        }
#elif MODE == 1
        // Per-lane varying index → LDC (per-thread) — NOT broadcast
        unsigned int base = (i + (threadIdx.x & 31) + u0);
        #pragma unroll
        for (int k = 0; k < NC; k++) {
            sum ^= CMEM[(base + k) & 255];
        }
#elif MODE == 2
        // Indirect inline asm — force ld.const.u32 with per-lane address that's actually uniform
        // Use shared-memory broadcast trick: get address into a per-thread register but identical
        unsigned int base = (i + u0);
        #pragma unroll
        for (int k = 0; k < NC; k++) {
            unsigned int v;
            // Inline ld.const.u32 — will become LDC.E or LDCU depending on uniformity analysis
            asm volatile("ld.const.u32 %0, [%1];"
                         : "=r"(v)
                         : "l"((unsigned long long)(uintptr_t)&CMEM[(base + k) & 255]));
            sum ^= v;
        }
#endif
    }

    if (sum == 0xDEADBEEFu) C[tid] = (float)sum;
}
