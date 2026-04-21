// G6: Cache modes for global loads
// .ca (cache all = L1+L2), .cg (cache global = L2 only, bypass L1), .cs (streaming = no cache)
// .lu (last use, evict from L1 after read)
// Test latency for each modifier with cold and warm cache
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // Setup pointer chain in A (single warp, modest size in L1)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int N = 32; // small, fits in L1
        for (unsigned int i = 0; i < N; i++) {
            ((unsigned int*)A)[i * 32] = ((i + 1) % N) * 32;
        }
    }
    __syncwarp();

    unsigned int idx = (threadIdx.x % 32) * 32;
    if ((unsigned)u2 != 0xDEADBEEFu) idx = 0;  // runtime dep

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned int x;
#if MODE == 0
        // Default LDG.E (no .ca/.cg/.cs hint)
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(x) : "l"(A + idx));
#elif MODE == 1
        // .ca = L1 cached
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(A + idx));
#elif MODE == 2
        // .cg = L2 only (bypass L1)
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(A + idx));
#elif MODE == 3
        // .cs = streaming (no cache)
        asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(x) : "l"(A + idx));
#elif MODE == 4
        // .lu = last use
        asm volatile("ld.global.lu.u32 %0, [%1];" : "=r"(x) : "l"(A + idx));
#elif MODE == 5
        // .nc = non-coherent (read-only via constant cache)
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(x) : "l"(A + idx));
#endif
        idx = x;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (idx == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = idx;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/load=%.2f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
