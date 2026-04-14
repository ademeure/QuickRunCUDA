// LDSM (ldmatrix) throughput in-depth — x1/x2/x4 + trans + fp16/bf16 widths.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Init shared memory with thread-unique values
    for (int k = threadIdx.x; k < 2048; k += BLOCK_SIZE) smem[k] = k + blockIdx.x * 17;
    __syncthreads();

    unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x & 0x1F) * 8]);
    unsigned int v0=0, v1=0, v2=0, v3=0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // x1 (1 matrix)
            unsigned int a;
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(a) : "r"(base));
            v0 ^= a;
#elif OP == 1  // x2 (2 matrices)
            unsigned int a,b;
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(a),"=r"(b) : "r"(base));
            v0 ^= a; v1 ^= b;
#elif OP == 2  // x4 (4 matrices, 128-bit / lane)
            unsigned int a,b,c,d;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(base));
            v0 ^= a; v1 ^= b; v2 ^= c; v3 ^= d;
#elif OP == 3  // x4.trans
            unsigned int a,b,c,d;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(base));
            v0 ^= a; v1 ^= b; v2 ^= c; v3 ^= d;
#elif OP == 4  // x1.trans
            unsigned int a;
            asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];" : "=r"(a) : "r"(base));
            v0 ^= a;
#elif OP == 5  // stmatrix.x1 — stores a matrix from register to shared
            asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};" :: "r"(base), "r"(v0 + j));
#elif OP == 6  // stmatrix.x4
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(base), "r"(v0 + j), "r"(v1+j), "r"(v2+j), "r"(v3+j));
#endif
        }
    }
    if ((int)(v0^v1^v2^v3) == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v0^v1^v2^v3;
}
