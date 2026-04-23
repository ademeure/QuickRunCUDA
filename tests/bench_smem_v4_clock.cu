// SMEM read peak — measure with clock64 inside kernel.
// Strategy: keep register pressure LOW. Use 4 chains, minimal state, large warp count.
// All loaded values feed back into addressing of subsequent loads (XOR cascade).

#ifndef N_LOADS
#define N_LOADS 4096
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int unused0, int seed, int unused2) {
    __shared__ unsigned int smem[2048];  // 8 KB
    #pragma unroll
    for (int i = threadIdx.x; i < 2048; i += BLOCK_SIZE) {
        smem[i] = i * 0x9E37u + threadIdx.x + 1;  // never zero
    }
    __syncthreads();

    // 4 independent chains, each carries the full v4 result forward
    unsigned int v0a = (threadIdx.x      ) & 511;
    unsigned int v0b = 0xA5A5A5A5;
    unsigned int v0c = 0xCAFEBABE;
    unsigned int v0d = 0xDEADBEEF;
    unsigned int v1a = (threadIdx.x + 17 ) & 511;
    unsigned int v1b = 0x12345678;
    unsigned int v1c = 0x87654321;
    unsigned int v1d = 0x55555555;
    unsigned int v2a = (threadIdx.x + 31 ) & 511;
    unsigned int v2b = 0xAAAAAAAA;
    unsigned int v2c = 0xCCCCCCCC;
    unsigned int v2d = 0x33333333;
    unsigned int v3a = (threadIdx.x + 47 ) & 511;
    unsigned int v3b = 0x99999999;
    unsigned int v3c = 0x66666666;
    unsigned int v3d = 0x77777777;

    unsigned int base = (unsigned)__cvta_generic_to_shared(smem);

    unsigned long long t0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int n = 0; n < N_LOADS; n++) {
        unsigned int a, b, c, d;
        // Address derived from full v4 of previous load -> compiler can't precompute
        unsigned int idx0 = (v0a ^ v0b ^ v0c ^ v0d ^ n) & 511;
        unsigned int idx1 = (v1a ^ v1b ^ v1c ^ v1d ^ n*3) & 511;
        unsigned int idx2 = (v2a ^ v2b ^ v2c ^ v2d ^ n*7) & 511;
        unsigned int idx3 = (v3a ^ v3b ^ v3c ^ v3d ^ n*11) & 511;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(base + idx0*16));
        v0a=a; v0b=b; v0c=c; v0d=d;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(base + idx1*16));
        v1a=a; v1b=b; v1c=c; v1d=d;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(base + idx2*16));
        v2a=a; v2b=b; v2c=c; v2d=d;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a),"=r"(b),"=r"(c),"=r"(d) : "r"(base + idx3*16));
        v3a=a; v3b=b; v3c=c; v3d=d;
    }

    unsigned long long t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    unsigned int acc = v0a^v0b^v0c^v0d^v1a^v1b^v1c^v1d^v2a^v2b^v2c^v2d^v3a^v3b^v3c^v3d;
    unsigned int tid_g = blockIdx.x * blockDim.x + threadIdx.x;
    ((unsigned int*)C)[tid_g * 4 + 0] = acc;
    ((unsigned int*)C)[tid_g * 4 + 1] = (unsigned int)(t1 - t0);
    ((unsigned int*)C)[tid_g * 4 + 2] = (unsigned int)((t1 - t0) >> 32);
    ((unsigned int*)C)[tid_g * 4 + 3] = N_LOADS * 4;
}
