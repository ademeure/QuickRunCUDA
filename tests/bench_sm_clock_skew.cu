extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (threadIdx.x != 0) return;
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    unsigned long long c0, g0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c0));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));

    // Make work non-DCE'able by writing acc to memory based on it
    unsigned acc = seed;
    #pragma unroll 1
    for (int i = 0; i < 4096; i++) {
        acc = acc * 1664525u + 1013904223u + (acc >> 17);
    }

    unsigned long long c1, g1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c1));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));

    unsigned long long* out = (unsigned long long*)C;
    unsigned idx = blockIdx.x;
    out[idx*4 + 0] = ((unsigned long long)smid << 32) | acc;  // pack smid + acc
    out[idx*4 + 1] = c1 - c0;
    out[idx*4 + 2] = g1 - g0;
    out[idx*4 + 3] = c0;
}
