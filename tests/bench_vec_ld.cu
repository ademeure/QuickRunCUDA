// Vector load/store throughput — proper dep chain, full TLP.

#ifndef UNROLL
#define UNROLL 8
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v0=0,v1=0,v2=0,v3=0;
    // Each thread reads its own stride-aligned region of global memory
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Offset hops through a large window to avoid cache hits
            unsigned long long off = ((tid * UNROLL + j) ^ i) & 0xFFFFFFu;
            unsigned long long ga = (unsigned long long)A + off * 16;
#if OP == 0  // ld.global.v4.u32 throughput
            unsigned int a,b,c,d;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "l"(ga));
            v0 ^= a; v1 ^= b; v2 ^= c; v3 ^= d;
#elif OP == 1  // ld.global.nc.v4.u32 (const-cached, read-only)
            unsigned int a,b,c,d;
            asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "l"(ga));
            v0 ^= a; v1 ^= b; v2 ^= c; v3 ^= d;
#elif OP == 2  // ld.global.u32 (scalar)
            unsigned int a;
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(a) : "l"(ga));
            v0 ^= a;
#elif OP == 3  // ld.shared.v4.u32
            unsigned int sbase = (unsigned)__cvta_generic_to_shared(&smem[((threadIdx.x + j) & 0x1FF) * 4]);
            unsigned int a,b,c,d;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(sbase));
            v0 ^= a; v1 ^= b; v2 ^= c; v3 ^= d;
#elif OP == 4  // ld.shared.u32 (scalar)
            unsigned int sbase = (unsigned)__cvta_generic_to_shared(&smem[((threadIdx.x + j) & 0x7FF)]);
            unsigned int a;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(a) : "r"(sbase));
            v0 ^= a;
#endif
        }
    }
    if ((int)(v0^v1^v2^v3) == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v0^v1^v2^v3;
}
