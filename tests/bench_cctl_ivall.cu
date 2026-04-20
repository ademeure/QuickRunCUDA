// Try various PTX patterns that might emit CCTL.IVALL
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int* p = (int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    int x = (int)u2;

#if MODE == 0
    // PTX cctl::ivall
    asm("cctl::ivall;");
#elif MODE == 1
    // Try specific cache control
    asm("cctl::wb;");
#elif MODE == 2
    // Variant scope
    asm("cctl::all.global.ivall;");
#elif MODE == 3
    // Per-line invalidation
    asm volatile("cctl.ivl.global [%0];" :: "l"(p + idx));
#elif MODE == 4
    // Per-line wb
    asm volatile("cctl.wb.global [%0];" :: "l"(p + idx));
#endif

    p[idx] = x;
}
