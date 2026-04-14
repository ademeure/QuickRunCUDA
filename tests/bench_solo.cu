// Generic solo op throughput tester
// OP_INSTR is a chunk of PTX; N_CHAINS parallel chains of U ops each per outer iter
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef N_CHAINS
#define N_CHAINS 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef OP_KIND
#define OP_KIND 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
#if OP_KIND == 0
    // POPC chains
    unsigned int r[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) r[k] = threadIdx.x + k + 0xCAFE1001u;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("popc.b32 %0, %0;" : "+r"(r[k]));
            }
        }
    }
    unsigned int acc = 0; for (int k = 0; k < N_CHAINS; k++) acc ^= r[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#elif OP_KIND == 1
    // SHFL chains
    unsigned int r[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) r[k] = threadIdx.x + k;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1f, 0xffffffff;" : "+r"(r[k]));
            }
        }
    }
    unsigned int acc = 0; for (int k = 0; k < N_CHAINS; k++) acc ^= r[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#elif OP_KIND == 2
    // F2F.F16.F32 chains
    float r[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) r[k] = (float)(threadIdx.x + k) * 0.25f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("{ .reg .b16 h; cvt.rn.f16.f32 h, %0; cvt.f32.f16 %0, h; }" : "+f"(r[k]));
            }
        }
    }
    unsigned int acc = 0; for (int k = 0; k < N_CHAINS; k++) acc ^= __float_as_int(r[k]);
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#elif OP_KIND == 3
    // I2F cvt.rn.f32.s32 + back
    float r[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) r[k] = (float)(threadIdx.x + k);
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("{ .reg .s32 t; cvt.rzi.s32.f32 t, %0; cvt.rn.f32.s32 %0, t; }" : "+f"(r[k]));
            }
        }
    }
    unsigned int acc = 0; for (int k = 0; k < N_CHAINS; k++) acc ^= __float_as_int(r[k]);
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#elif OP_KIND == 4
    // VOTE ballot
    unsigned int r[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) r[k] = threadIdx.x + k;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned int v;
                asm volatile("vote.sync.ballot.b32 %0, 1, 0xffffffff;" : "=r"(v));
                r[k] ^= v;
            }
        }
    }
    unsigned int acc = 0; for (int k = 0; k < N_CHAINS; k++) acc ^= r[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#endif
}
