// Extended ldmatrix variants: FP8/FP6 LDSM + new Blackwell shapes.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef ITERS
#define ITERS 1024
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    for (int i = threadIdx.x; i < 2048; i += BLOCK_SIZE) smem[i] = i + blockIdx.x * 17;
    __syncthreads();
    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x & 0x1F) * 8]);
    unsigned v0=0,v1=0,v2=0,v3=0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
#if OP == 0  // x4 b16 (standard)
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(base));
#elif OP == 1  // x4.trans b16
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(base));
#elif OP == 2  // x8 (if exists) - try larger count
            unsigned v4,v5,v6,v7;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(base));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                : "=r"(v4),"=r"(v5),"=r"(v6),"=r"(v7) : "r"(base+8));
            v0 ^= v4; v1 ^= v5; v2 ^= v6; v3 ^= v7;
#elif OP == 3  // b8x16.b6x16_p32 FP6 LDSM (new Blackwell)
            asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 {%0,%1,%2,%3}, [%4];"
                : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(base));
#elif OP == 4  // b8x16.b4x16_p64 FP4 LDSM (Blackwell FP4)
            asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];"
                : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(base));
#endif
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = v0^v1^v2^v3;
    }
}
