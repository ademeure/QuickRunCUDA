// V8: HMMA.F16 tensor peak via mma.sync.m16n8k16.f16.f16
// Each instruction: 16*8*16 = 2048 FP16 FMAs = 4096 FLOPs per warp
// Theoretical: depends on issue rate — typically 1/8 cycles per SMSP for Blackwell FP16 MMA
//
// Kernel uses self-feeding accumulator (anti-DCE) and 8 chains for ILP.

#ifndef N_CHAINS
#define N_CHAINS 8
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Each warp holds:
    //   A fragment: 4 × b32 (8 halves) per thread — 16×16 FP16 matrix
    //   B fragment: 2 × b32 (4 halves) per thread — 16×8 FP16 matrix
    //   C/D fragment (FP16 accumulator): 2 × b32 (4 halves) per thread — 16×8 matrix

    // Init with varying seeds to defeat common-value optimization
    unsigned a0 = (unsigned)(threadIdx.x * 17 + seed);
    unsigned a1 = (unsigned)(threadIdx.x * 23 + seed + 1);
    unsigned a2 = (unsigned)(threadIdx.x * 29 + seed + 2);
    unsigned a3 = (unsigned)(threadIdx.x * 31 + seed + 3);
    unsigned b0 = (unsigned)(threadIdx.x * 37 + seed + 4);
    unsigned b1 = (unsigned)(threadIdx.x * 41 + seed + 5);

    // N_CHAINS independent accumulators
    unsigned c0[N_CHAINS], c1[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        c0[k] = (unsigned)(threadIdx.x + k);
        c1[k] = (unsigned)(threadIdx.x + k + 100);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                " {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
                : "+r"(c0[k]), "+r"(c1[k])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1)
            );
        }
    }

    // Anti-DCE
    unsigned acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= c0[k] ^ c1[k];
    if (acc == 0xFFFFFFFFu) C[blockIdx.x * blockDim.x + threadIdx.x] = (float)acc;
}
