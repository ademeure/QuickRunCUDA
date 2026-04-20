// SETP (predicate set) throughput on B300.
// Tests: ISETP (integer compare), FSETP (float compare), and chains
// Each variant compares two values, sets a predicate, then we use the
// predicate (e.g., predicate-add or predicate-store) to defeat DCE.

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP_MODE
#define OP_MODE 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS], b[N_CHAINS];
    float fv[N_CHAINS], fb[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (unsigned)(threadIdx.x * 131 + k * 17);
        b[k] = (unsigned)(threadIdx.x * 271 + k * 23);
        fv[k] = (float)(threadIdx.x + k);
        fb[k] = (float)(threadIdx.x * 2 + k);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if OP_MODE == 0
                // ISETP.GT + selp (conditional select via predicate)
                unsigned int sel;
                asm volatile("{ .reg .pred p; setp.gt.u32 p, %1, %2; selp.b32 %0, %1, %2, p; }"
                             : "=r"(sel) : "r"(v[k]), "r"(b[k]));
                v[k] = sel ^ (sel >> 1);
#elif OP_MODE == 1
                // FSETP.GT + selp
                float sel;
                asm volatile("{ .reg .pred p; setp.gt.f32 p, %1, %2; selp.f32 %0, %1, %2, p; }"
                             : "=f"(sel) : "f"(fv[k]), "f"(fb[k]));
                fv[k] = sel * 31.0f + fb[k];
#elif OP_MODE == 2
                // ISETP only, no selp - use to mask via @p add
                asm volatile("{ .reg .pred p; setp.gt.u32 p, %0, %1; @p add.u32 %0, %0, 1; }"
                             : "+r"(v[k]) : "r"(b[k]));
#elif OP_MODE == 3
                // ISETP and-fold (chained predicates)
                asm volatile("{ .reg .pred p,q; setp.gt.u32 p, %0, %1; setp.lt.and.u32 q, %0, %1, p; @q add.u32 %0, %0, 1; }"
                             : "+r"(v[k]) : "r"(b[k]));
#elif OP_MODE == 4
                // Reference: pure IADD3 (already tested)
                asm volatile("add.u32 %0, %0, %1;" : "+r"(v[k]) : "r"(b[k]));
#endif
            }
        }
    }

    unsigned int acc = 0; float facc = 0.0f;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) { acc ^= v[k]; facc += fv[k]; }
    if ((acc == (unsigned)seed) && ((int)facc == seed))
        ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
