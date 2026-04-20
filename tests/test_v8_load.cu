extern "C" __global__ void kernel(float* A, float* B, float* C, int iters, int u1, int u2) {
    unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned* base = (unsigned*)A;
    unsigned acc = 0;
    for (int i = 0; i < iters; i++) {
        unsigned offset = tid * 8 + (i % 4) * 256;  // bounded
        uint4 v0;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                     : "l"(&base[offset]));
        acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w;
    }
    if (acc == 0xDEADBEEF) ((unsigned*)C)[tid] = acc;
}
