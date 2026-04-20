#ifndef MODE
#define MODE 0
#endif
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int* p = (int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    int x;
#if MODE == 0
    x = p[idx];
#elif MODE == 1
    asm("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(x) : "l"(p + idx));
#elif MODE == 2
    asm("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(x) : "l"(p + idx));
#elif MODE == 3
    asm("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(x) : "l"(p + idx));
#elif MODE == 4
    asm volatile("membar.gl;\n\t ld.global.s32 %0, [%1];" : "=r"(x) : "l"(p + idx));
#endif
    p[idx] = x;
}
