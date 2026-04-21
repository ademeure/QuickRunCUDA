// ldmatrix vs explicit LDS for tensor fragment load
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    __shared__ unsigned int smem[32 * 8];

    if (threadIdx.x < 32) {
        for (int i = threadIdx.x; i < 32 * 8; i += 32) smem[i] = i ^ (unsigned)u2;
    }
    __syncwarp();

    unsigned int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    unsigned int addr = (unsigned)__cvta_generic_to_shared(smem) + (threadIdx.x & 7) * 32;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // ldmatrix x4: warp loads 4 8x8 b16 tiles in one inst
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#elif MODE == 1
        // Explicit LDS x4 (4 LDS.32 per thread)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#elif MODE == 2
        // Scalar LDS x4 separately
        asm volatile("ld.shared.u32 %0, [%4];"
                     "ld.shared.u32 %1, [%4 + 4];"
                     "ld.shared.u32 %2, [%4 + 8];"
                     "ld.shared.u32 %3, [%4 + 12];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#endif
        addr ^= ((unsigned)u2 * r0);  // chain dep
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (r0 == (unsigned)seed) C[blockIdx.x] = (float)(r0 + r1 + r2 + r3);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
