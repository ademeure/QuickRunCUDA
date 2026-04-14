// Focused shift/rotate catalog.
// Shift amounts use runtime data to defeat DCE.

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
                unsigned int nxt = v[(k+1) & (N_CHAINS-1)];  // runtime value for shift amt
#if OP == 50  // SHL data-dep amount
                asm volatile("shl.b32 %0, %0, %1;" : "+r"(x) : "r"(nxt & 0x1Fu));
#elif OP == 51  // SHR (logical, u32)
                asm volatile("shr.u32 %0, %0, %1;" : "+r"(x) : "r"(nxt & 0x1Fu));
#elif OP == 52  // SHR arith s32
                int ix = (int)x;
                asm volatile("shr.s32 %0, %0, %1;" : "+r"(ix) : "r"(nxt & 0x1Fu));
                x = (unsigned int)ix;
#elif OP == 53  // SHF.L WRAP
                asm volatile("shf.l.wrap.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(nxt), "r"(nxt & 0x1Fu));
#elif OP == 54  // SHF.R WRAP
                asm volatile("shf.r.wrap.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(nxt), "r"(nxt & 0x1Fu));
#elif OP == 55  // SHF.L CLAMP
                asm volatile("shf.l.clamp.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(nxt), "r"(nxt & 0x1Fu));
#elif OP == 56  // SHF.R CLAMP
                asm volatile("shf.r.clamp.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(nxt), "r"(nxt & 0x1Fu));
#elif OP == 57  // PRMT control-variable
                asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(x) : "r"(nxt), "r"(nxt & 0xFFFFu));
#elif OP == 58  // ROTATE via SHF.L.WRAP with same src
                asm volatile("shf.l.wrap.b32 %0, %0, %0, %1;" : "+r"(x) : "r"(nxt & 0x1Fu));
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
