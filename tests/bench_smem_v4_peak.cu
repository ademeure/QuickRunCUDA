// SMEM read peak via ld.shared.v4.u32 — chain-dependent to defeat DCE.
//
// Each thread loads 16 B per ld.shared.v4.u32 instruction.
// Throughput target: 128 B/clk/SM × 148 SMs × clock.
//   At 1920 MHz: 128 × 148 × 1.92e9 = 36.4 TB/s
//   At 2032 MHz: 128 × 148 × 2.032e9 = 38.5 TB/s

#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    // 16 KiB / block static smem — small enough to fit 2 CTAs/SM (148*2=296 CTAs total)
    __shared__ unsigned int smem[4096];
    // Init smem with unique values so loads cannot be folded
    #pragma unroll
    for (int i = threadIdx.x; i < 4096; i += BLOCK_SIZE) {
        smem[i] = i * 0x9E3779B1u + 0xDEADBEEFu;
    }
    __syncthreads();

    // N_CHAINS independent dependency chains => ILP without serializing
    unsigned int v[N_CHAINS][4];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k][0] = threadIdx.x + k;
        v[k][1] = threadIdx.x ^ (k+1);
        v[k][2] = threadIdx.x + k*7;
        v[k][3] = threadIdx.x ^ (k*13);
    }

    // Each chain has its own base offset; addresses depend on prior loaded values
    // but we mask back into [0, 4095) so we stay in-bounds.
    unsigned int base[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        base[k] = (unsigned)__cvta_generic_to_shared(smem);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                // Address depends on prior v[k][0] -> dep chain prevents DCE
                // 1024 v4 slots × 16B = 16384 B = exactly smem size; mask 0x3FF, last addr 1023*16 = 16368, +12 = 16380 < 16384 OK
                unsigned int idx = (v[k][0] + j*131 + k*17) & 1023;
                unsigned int addr = base[k] + idx * 16;
                unsigned int a, b, c, d;
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
                v[k][0] = a;
                v[k][1] ^= b;
                v[k][2] ^= c;
                v[k][3] ^= d;
            }
        }
    }
    // Unconditional store of folded results to defeat DCE
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        acc ^= v[k][0] ^ v[k][1] ^ v[k][2] ^ v[k][3];
    }
    if ((int)acc == seed) {
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    }
}
