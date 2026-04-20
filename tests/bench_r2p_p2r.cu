// R2P/P2R cost: predicate-to-register and reverse.
// PTX: setp -> @p add (predicated) is the natural form
// To convert pred to reg: `selp.b32 r, 1, 0, p;` then back via `setp.ne.b32 p, r, 0;`
//
// Mode 0: setp + selp (P->R then unused)
// Mode 1: setp + selp + setp (P->R->P round-trip)
// Mode 2: vote.ballot.sync (warp-wide P->R32)
// Mode 3: __ballot_sync (high-level wrapper)

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS], b[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (unsigned)(threadIdx.x * 131 + k * 17 + (unsigned)u2);
        b[k] = (unsigned)(threadIdx.x * 271 + k * 23);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
#if MODE == 0
            // setp + selp (P->R)
            asm volatile("{ .reg .pred p; setp.gt.u32 p, %0, %1; selp.b32 %0, 1, 0, p; }"
                         : "+r"(v[k]) : "r"(b[k]));
#elif MODE == 1
            // P -> R -> P round-trip
            asm volatile("{ .reg .pred p; .reg .b32 r; "
                         "setp.gt.u32 p, %0, %1; selp.b32 r, 1, 0, p; "
                         "setp.ne.b32 p, r, 0; @p add.u32 %0, %0, 1; }"
                         : "+r"(v[k]) : "r"(b[k]));
#elif MODE == 2
            // PTX vote.ballot.sync (P -> R32 warp-wide)
            unsigned int mask;
            asm volatile("{ .reg .pred p; setp.gt.u32 p, %1, %2; vote.sync.ballot.b32 %0, p, 0xFFFFFFFF; }"
                         : "=r"(mask) : "r"(v[k]), "r"(b[k]));
            v[k] ^= mask;
#elif MODE == 3
            // CUDA intrinsic
            unsigned int mask = __ballot_sync(0xFFFFFFFFu, v[k] > b[k]);
            v[k] ^= mask;
#endif
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
