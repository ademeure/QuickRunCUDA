// shfl variant throughput
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned int v = threadIdx.x + (unsigned)u2;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        v = __shfl_sync(0xFFFFFFFF, v, 0);  // broadcast lane 0
#elif MODE == 1
        v = __shfl_xor_sync(0xFFFFFFFF, v, 1);  // butterfly
#elif MODE == 2
        v = __shfl_up_sync(0xFFFFFFFF, v, 1);
#elif MODE == 3
        v = __shfl_down_sync(0xFFFFFFFF, v, 1);
#elif MODE == 4
        // shfl with dynamic lane (worst case)
        v = __shfl_sync(0xFFFFFFFF, v, v & 31);
#endif
    }

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
}
