// V10: SHFL modes — idx/up/down/xor latency comparison
#ifndef MODE
#define MODE 0  // 0=idx, 1=up, 2=down, 3=xor (bfly)
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)threadIdx.x + (unsigned)seed;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    __syncwarp();

    #pragma unroll 16
    for (int i = 0; i < CHAIN_LEN; i++) {
#if MODE == 0
        // SHFL.IDX — get from absolute lane
        v = __shfl_sync(0xFFFFFFFF, v, (v + 1) & 31);
#elif MODE == 1
        // SHFL.UP — get from lower lane (offset)
        v = __shfl_up_sync(0xFFFFFFFF, v, 1);
#elif MODE == 2
        // SHFL.DOWN — get from higher lane
        v = __shfl_down_sync(0xFFFFFFFF, v, 1);
#elif MODE == 3
        // SHFL.BFLY — XOR pattern
        v = __shfl_xor_sync(0xFFFFFFFF, v, 1);
#endif
    }

    __syncwarp();
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned*)C)[2] = v;
    }
}
