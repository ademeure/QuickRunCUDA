// Clean u32 ADD test — settle whether peak = 64 or 128 SASS/SM/cy.
// OP 0: simple 2-input `add.u32`
// OP 1: explicit PTX IADD3-like ternary `add.u32 d, a, b + c` approximated
// OP 2: force IADD3 via inline hand-written lop3-like pattern

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) v[k] = 0xDEADBEEFu ^ (threadIdx.x * 131 + k * 17);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned int x = v[k];
                unsigned int n1 = v[(k+1) & (N_CHAINS-1)];
                unsigned int n2 = v[(k+2) & (N_CHAINS-1)];
#if OP == 0  // simple 2-input add
                asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(n1));
#elif OP == 1  // two separate adds per loop iter (test if compiler fuses)
                asm volatile("add.u32 %0, %0, %1; add.u32 %0, %0, %2;" : "+r"(x) : "r"(n1), "r"(n2));
#elif OP == 2  // Explicit 3-input IADD3 via ptx
                asm volatile("{.reg .u32 t; add.u32 t, %1, %2; add.u32 %0, %0, t;}" : "+r"(x) : "r"(n1), "r"(n2));
#elif OP == 3  // Force SASS IADD3 directly (if nvcc exposes it via add3)
                asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(n1 + n2));  // compiler may emit IADD3
#endif
                v[k] = x;
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
