// L2 read peak with per-iter varying address + unconditional output (DCE-safe).
// WS_BYTES selects working set size; small WS (8-32 MB) should fit in one L2 side.
// .cg cache modifier = bypass L1, go straight to L2.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef UNROLL
#define UNROLL 16
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int ITERS, int seed, int WS_BYTES) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = (unsigned)(WS_BYTES - 1);
    unsigned acc0=0, acc1=0, acc2=0, acc3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned off = ((tid * 16 + (i + j) * 16 * gridDim.x * blockDim.x) & mask);
            unsigned long long addr = (unsigned long long)A + off;
            unsigned x0,x1,x2,x3;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    // UNCONDITIONAL output — no `if` guard
    C[tid] = acc0 ^ acc1 ^ acc2 ^ acc3;
}
