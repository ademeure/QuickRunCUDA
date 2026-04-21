// H8: SMEM read vs write vs LDS+STS pair power per byte
// MODE 0: pure LDS read (32 threads × 4B per iter × N inner)
// MODE 1: pure STS write
// MODE 2: LDS + STS pair
// MODE 3: STS.128 (16-byte vector store)
// MODE 4: LDS.128 (16-byte vector load)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(16) unsigned int smem[2048];
    if (threadIdx.x < 32) {
        for (int i = threadIdx.x; i < 2048; i += 32) smem[i] = i + (unsigned)u2;
    }
    __syncthreads();

    unsigned int smem_addr = __cvta_generic_to_shared(smem);
    unsigned int v = (unsigned)(threadIdx.x ^ u2);
    unsigned int sink = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 32
        for (int u = 0; u < 32; u++) {
            unsigned int off = ((threadIdx.x + i + u) & 511) * 4;
#if MODE == 0
            // Pure LDS
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr + off));
            sink ^= x;
#elif MODE == 1
            // Pure STS
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_addr + off), "r"(v));
            v += i;
#elif MODE == 2
            // LDS + STS
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr + off));
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_addr + off + 1024), "r"(x));
            sink ^= x;
#elif MODE == 3
            // STS.128 (vector 16B store)
            unsigned int off16 = ((threadIdx.x + i + u) & 127) * 16;
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
                :: "r"(smem_addr + off16), "r"(v), "r"(v+1), "r"(v+2), "r"(v+3));
            v += i;
#elif MODE == 4
            // LDS.128 (vector 16B load)
            unsigned int off16 = ((threadIdx.x + i + u) & 127) * 16;
            unsigned int x0,x1,x2,x3;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "r"(smem_addr + off16));
            sink ^= x0 ^ x1 ^ x2 ^ x3;
#endif
        }
    }

    if (sink == (unsigned)seed && v == (unsigned)seed) C[blockIdx.x] = (float)v;
}
