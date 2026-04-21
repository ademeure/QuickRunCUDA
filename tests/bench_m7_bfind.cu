// M7: bfind / brev / popc / clz throughput (bit ops on B300)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)(threadIdx.x ^ u2) | 1;  // ensure non-zero
    unsigned int x = 0xDEADBEEFu ^ (unsigned)u2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // bfind.u32 (find msb)
            unsigned int r;
            asm volatile("bfind.u32 %0, %1;" : "=r"(r) : "r"(v));
            v = r ^ x;
#elif MODE == 1
            // bfind.shiftamt (msb shift amount, 31-pos)
            unsigned int r;
            asm volatile("bfind.shiftamt.u32 %0, %1;" : "=r"(r) : "r"(v));
            v = r ^ x;
#elif MODE == 2
            // popc (population count)
            unsigned int r;
            asm volatile("popc.b32 %0, %1;" : "=r"(r) : "r"(v));
            v = r ^ x;
#elif MODE == 3
            // clz (count leading zeros)
            unsigned int r;
            asm volatile("clz.b32 %0, %1;" : "=r"(r) : "r"(v));
            v = r ^ x;
#elif MODE == 4
            // brev (bit reverse)
            unsigned int r;
            asm volatile("brev.b32 %0, %1;" : "=r"(r) : "r"(v));
            v = r ^ x;
#elif MODE == 5
            // bfe (bit field extract)
            unsigned int r;
            asm volatile("bfe.u32 %0, %1, 4, 8;" : "=r"(r) : "r"(v));
            v = r ^ x;
#elif MODE == 6
            // baseline LOP3 (Cluster B reference)
            asm volatile("lop3.b32 %0, %0, %1, %1, 0xC8;" : "+r"(v) : "r"(x));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/op=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS/16.0);
    }
}
