// Pick N SMs by smid list and measure per-SM fence cost.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 200
#endif
#ifndef SMID_MASK_LO
#define SMID_MASK_LO 0x3  // bits 0-63 of which smids to activate
#endif
#ifndef SMID_MASK_HI
#define SMID_MASK_HI 0x0
#endif
#ifndef SMID_MASK_128
#define SMID_MASK_128 0x0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    // Check if my smid is in the mask
    unsigned long long mlo = SMID_MASK_LO;
    unsigned long long mhi = SMID_MASK_HI;
    unsigned long long m128 = SMID_MASK_128;  // for smids 128-147
    bool active = false;
    if (smid < 64) active = (mlo >> smid) & 1;
    else if (smid < 128) active = (mhi >> (smid - 64)) & 1;
    else if (smid < 148) active = (m128 >> (smid - 128)) & 1;
    if (!active) return;
    if (threadIdx.x != 0) return;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid * 4;

    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        *(volatile unsigned*)my_addr = i + seed;
        asm volatile("fence.sc.sys;");
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    atomicCAS(C + smid, 0, total / ITERS);
}
