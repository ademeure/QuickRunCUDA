// D8: Address generation pipeline depth
// Test: how many independent LDS loads can be in-flight before AGEN stalls?
// Vary chain length of dependent address calculations
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
    unsigned int v = (unsigned)(threadIdx.x ^ u2);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // Pre-computed address (no AGEN dependency on prior load)
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base_addr + ((u + i) & 1023) * 4));
            v ^= x;
#elif MODE == 1
            // Loaded value used as address (address-bound: AGEN depends on previous LDS)
            unsigned int off = (v & 1023) * 4;
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base_addr + off));
            v = x;  // chain: v depends on this load fully
#elif MODE == 2
            // 2 independent loads then chain
            unsigned int x1, x2;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x1) : "r"(base_addr + ((u + i) & 1023) * 4));
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x2) : "r"(base_addr + ((u + i + 17) & 1023) * 4));
            v ^= x1 ^ x2;
#elif MODE == 3
            // Chain of 4 dependent address computations (each depends on prev result for addr)
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base_addr + ((v + 0) & 1023) * 4));
            v = x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base_addr + ((v + 1) & 1023) * 4));
            v = x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base_addr + ((v + 2) & 1023) * 4));
            v = x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base_addr + ((v + 3) & 1023) * 4));
            v = x;
#elif MODE == 4
            // Pre-computed address with 4 IADD3 ops "before" the load
            unsigned int off = (u + i) & 1023;
            off = off + 1;
            off = off + 2;
            off = off + 3;
            off = off & 1023;
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base_addr + off * 4));
            v ^= x;
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f cy/op=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/16.0);
    }
}
