// FFMA + LDS load overlap
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ unsigned int smem[1024];
    for (int i = threadIdx.x; i < 1024; i += 32) smem[i] = i + (unsigned)u2;
    __syncwarp();

    unsigned int base_addr = __cvta_generic_to_shared(smem);
    float x = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float y = (float)threadIdx.x * 0.002f + 1.0f;
    float z = 0.5f;
    unsigned int v = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned int off = (i & 1023) * 4;
#if MODE == 0
        // Pure LDS load (independent addresses, no chain)
        unsigned int x1;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x1) : "r"(base_addr + off));
        v ^= x1;
#elif MODE == 1
        // Pure 4 FFMA (chained)
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
#elif MODE == 2
        // 1 LDS + 4 FFMA — should overlap
        unsigned int x1;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x1) : "r"(base_addr + off));
        v ^= x1;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
#elif MODE == 3
        // 1 LDS + 8 FFMA
        unsigned int x1;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x1) : "r"(base_addr + off));
        v ^= x1;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
#elif MODE == 4
        // 1 LDS + 16 FFMA
        unsigned int x1;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x1) : "r"(base_addr + off));
        v ^= x1;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
#elif MODE == 5
        // 4 LDS + 16 FFMA
        unsigned int x1, x2, x3, x4;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x1) : "r"(base_addr + off));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x2) : "r"(base_addr + off + 4));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x3) : "r"(base_addr + off + 8));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x4) : "r"(base_addr + off + 12));
        v ^= x1 ^ x2 ^ x3 ^ x4;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)x == seed && v == (unsigned)seed) C[blockIdx.x] = x + (float)v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.2f\n", MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
