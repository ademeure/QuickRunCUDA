// Warp-reduce method comparison: redux.sync.add vs SHFL.bfly chain.
// Mode 0: 5-step SHFL.bfly (XOR butterfly) + 5 ADDs
// Mode 1: redux.sync.add (single inst, Hopper+)
// Mode 2: 5-step SHFL.up
// Mode 3: explicit __shfl_xor_sync wrappers (compiler-emitted)
//
// Each thread reduces its register; result lands in all lanes (mode 0/3) or
// lane 0 (modes 1/2 specifics may vary).

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef METHOD
#define METHOD 0
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++)
        v[k] = (unsigned)(threadIdx.x * 131 + k * 17);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned int x = v[k];
#if METHOD == 0
                // 5-step SHFL.bfly chain
                unsigned int y;
                asm volatile("shfl.sync.bfly.b32 %0, %1, 16, 0x1f, 0xffffffff;" : "=r"(y) : "r"(x)); x += y;
                asm volatile("shfl.sync.bfly.b32 %0, %1,  8, 0x1f, 0xffffffff;" : "=r"(y) : "r"(x)); x += y;
                asm volatile("shfl.sync.bfly.b32 %0, %1,  4, 0x1f, 0xffffffff;" : "=r"(y) : "r"(x)); x += y;
                asm volatile("shfl.sync.bfly.b32 %0, %1,  2, 0x1f, 0xffffffff;" : "=r"(y) : "r"(x)); x += y;
                asm volatile("shfl.sync.bfly.b32 %0, %1,  1, 0x1f, 0xffffffff;" : "=r"(y) : "r"(x)); x += y;
#elif METHOD == 1
                // redux.sync.add — single inst warp reduce (Hopper/Blackwell)
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(x) : "r"(x));
#elif METHOD == 2
                // 5-step SHFL.up chain (different topology)
                unsigned int y;
                asm volatile("shfl.sync.up.b32 %0, %1,  1, 0, 0xffffffff;" : "=r"(y) : "r"(x)); x += y;
                asm volatile("shfl.sync.up.b32 %0, %1,  2, 0, 0xffffffff;" : "=r"(y) : "r"(x)); x += y;
                asm volatile("shfl.sync.up.b32 %0, %1,  4, 0, 0xffffffff;" : "=r"(y) : "r"(x)); x += y;
                asm volatile("shfl.sync.up.b32 %0, %1,  8, 0, 0xffffffff;" : "=r"(y) : "r"(x)); x += y;
                asm volatile("shfl.sync.up.b32 %0, %1, 16, 0, 0xffffffff;" : "=r"(y) : "r"(x)); x += y;
#elif METHOD == 3
                // CUDA intrinsic __shfl_xor_sync chain
                x += __shfl_xor_sync(0xffffffff, x, 16);
                x += __shfl_xor_sync(0xffffffff, x, 8);
                x += __shfl_xor_sync(0xffffffff, x, 4);
                x += __shfl_xor_sync(0xffffffff, x, 2);
                x += __shfl_xor_sync(0xffffffff, x, 1);
#endif
                v[k] = x ^ v[k];  // chain
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)ITERS * (unsigned long long)N_CHAINS;
        printf("METHOD=%d iters=%d N_CHAINS=%d clk=%llu cy/reduce=%.3f\n",
               METHOD, ITERS, N_CHAINS, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
