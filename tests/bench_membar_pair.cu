// Select 2 SMs by smid pair — same TPC (0,1), same GPC-different-TPC (0,2), or diff GPC (0,16).
// Only those 2 SMs fence; others exit.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef SMID_A
#define SMID_A 0
#endif
#ifndef SMID_B
#define SMID_B 1
#endif
#ifndef ITERS
#define ITERS 200
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (smid != SMID_A && smid != SMID_B) return;
    if (threadIdx.x != 0) return;
    // Use smid as slot index in C
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid * 16;

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
    // Store in slot[smid]; if multiple blocks land on same SM only first wins
    atomicCAS(C + smid, 0, total / ITERS);
}
