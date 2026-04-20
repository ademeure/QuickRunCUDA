// Cache line size inference: walk DRAM with increasing strides, time per element.
// If line size = 128B (32 ints), then stride < 32 amortizes line fetch over multiple
// elements; stride >= 32 pays full line fetch per element.
//
// Single warp loads N elements with stride S, all distinct cache lines after S=32.

#ifndef STRIDE_BYTES
#define STRIDE_BYTES 4
#endif
#ifndef N_LOADS
#define N_LOADS 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int* Au = (unsigned int*)A;
    int stride_words = STRIDE_BYTES / 4;

    unsigned int acc = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_LOADS; i++) {
        // Each lane reads a unique address with given stride
        unsigned int idx = (i * 32 * stride_words + threadIdx.x * stride_words +
                           (unsigned)u2 * acc) & 0x3FFFFFF;  // 64M dwords = 256MB
        unsigned int x;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(Au + idx));
        acc ^= x;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("STRIDE_BYTES=%d N_LOADS=%d clk=%llu cy/load=%.3f\n",
               STRIDE_BYTES, N_LOADS, t1 - t0, (double)(t1-t0)/(double)N_LOADS);
    }
}
