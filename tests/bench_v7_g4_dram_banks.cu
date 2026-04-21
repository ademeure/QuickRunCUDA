// V7 G4: DRAM bank conflict test
// Each thread reads from a base address with stride; vary stride
// MODE 0: stride 4 KB (one HBM3E page boundary)
// MODE 1: stride 64 KB (multi-page)
// MODE 2: stride 256 B (sub-page)
// All threads in warp access same bank → bank conflict; different banks → BW
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(int* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

#if MODE == 0
    int STRIDE = 4096 / 4;  // 1024 ints per thread offset
#elif MODE == 1
    int STRIDE = 65536 / 4;  // 16384 ints
#elif MODE == 2
    int STRIDE = 256 / 4;  // 64 ints
#endif

    unsigned int MASK = (1024 * 1024 * 256 / 4) - 1;  // 256 MB

    int sum = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Each thread reads from gtid + i with stride between threads
        unsigned int idx = (threadIdx.x * STRIDE + i * 32) & MASK;
        sum += A[idx];
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (sum == 0xCAFEBABE) C[blockIdx.x] = (float)sum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d STRIDE=%d cy/access=%.2f\n",
               MODE, STRIDE, (double)(t1-t0)/(double)ITERS);
    }
}
