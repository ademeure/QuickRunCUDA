// atomic on volatile pointer: SASS difference?
// Mode 0: atomicAdd on regular int*
// Mode 1: atomicAdd on volatile int* (cast)
// Mode 2: atomicAdd on volatile + acquire fence
// Mode 3: atomicAdd_system (cross-device atom)

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int* p = (int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;

#if MODE == 0
    atomicAdd(p + idx, 1);
#elif MODE == 1
    atomicAdd((volatile int*)(p + idx), 1);  // wait no, atomicAdd doesn't take volatile
    // Actually, atomicAdd on volatile* requires PTX
    asm volatile("atom.global.add.u32 %0, [%1], 1;" : "=r"(p[idx]) : "l"(p + idx));
#elif MODE == 2
    asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], 1;" : "=r"(p[idx]) : "l"(p + idx));
#elif MODE == 3
    atomicAdd_system(p + idx, 1);
#endif
}
