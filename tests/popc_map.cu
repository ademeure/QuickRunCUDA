extern "C" __global__ void kernel(float* A, float* B, float* C, int a0, int a1, int a2) {
    unsigned int x = 0xDEADBEEF ^ threadIdx.x;
    unsigned int y;
    asm volatile("popc.b32 %0, %1;" : "=r"(y) : "r"(x));
    asm volatile("brev.b32 %0, %1;" : "=r"(y) : "r"(y));
    asm volatile("clz.b32 %0, %1;" : "=r"(y) : "r"(y));
    asm volatile("bfind.u32 %0, %1;" : "=r"(y) : "r"(y));
    if (a0 == 12345) ((unsigned int*)C)[threadIdx.x] = y;
}
