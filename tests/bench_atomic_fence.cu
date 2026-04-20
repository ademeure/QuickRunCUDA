// Atomic with fence/scope variants - SASS comparison.

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int* p = (int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    unsigned int v = (unsigned)idx + (unsigned)u2;

#if MODE == 0
    atomicAdd(p + idx, (int)v);
#elif MODE == 1
    asm volatile("atom.relaxed.gpu.global.add.u32 _, [%0], %1;" :: "l"(p + idx), "r"(v));
#elif MODE == 2
    asm volatile("atom.acquire.gpu.global.add.u32 _, [%0], %1;" :: "l"(p + idx), "r"(v));
#elif MODE == 3
    asm volatile("atom.release.gpu.global.add.u32 _, [%0], %1;" :: "l"(p + idx), "r"(v));
#elif MODE == 4
    asm volatile("atom.acq_rel.gpu.global.add.u32 _, [%0], %1;" :: "l"(p + idx), "r"(v));
#elif MODE == 5
    // Explicit fence + atom (standalone)
    asm volatile("membar.gpu;");
    asm volatile("atom.global.add.u32 _, [%0], %1;" :: "l"(p + idx), "r"(v));
#elif MODE == 6
    // atom + fence
    asm volatile("atom.global.add.u32 _, [%0], %1;" :: "l"(p + idx), "r"(v));
    asm volatile("membar.gpu;");
#endif
}
