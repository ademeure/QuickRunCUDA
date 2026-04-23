// SMEM read throughput — independent loads, loop-counter-derived addresses,
// ALL loaded values mandatorily feed forward into output.

#ifndef N_ILP
#define N_ILP 16
#endif
#ifndef N_LOADS_OUTER
#define N_LOADS_OUTER 512
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int unused0, int seed, int unused2) {
    __shared__ unsigned int smem[2048];
    #pragma unroll
    for (int i = threadIdx.x; i < 2048; i += BLOCK_SIZE) {
        smem[i] = i * 0x9E37u + threadIdx.x + 1;
    }
    __syncthreads();

    unsigned int base = (unsigned)__cvta_generic_to_shared(smem);

    // Running accumulator across ALL loaded values. Store final to C.
    unsigned int acc_a = 0, acc_b = 0, acc_c = 0, acc_d = 0;

    unsigned long long t0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int n = 0; n < N_LOADS_OUTER; n++) {
        // Address for ILP slot k = loop-counter derived; won't reduce to const
        // because n varies at runtime.
        unsigned int a[N_ILP], b[N_ILP], c[N_ILP], d[N_ILP];
        #pragma unroll
        for (int k = 0; k < N_ILP; k++) {
            // Each thread hits a slot based on (threadIdx.x + k*stride + n*offset) mod 512
            unsigned int idx = ((threadIdx.x * 7) + k * 37 + n * 11) & 511;
            unsigned int addr = base + idx * 16;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a[k]), "=r"(b[k]), "=r"(c[k]), "=r"(d[k]) : "r"(addr));
        }
        // Merge all loaded values into accumulator. `acc_a >> 16` used in next iter's idx
        // creates a weak dependency but only 1 bit of info per iter — doesn't fold.
        #pragma unroll
        for (int k = 0; k < N_ILP; k++) {
            acc_a ^= a[k];
            acc_b ^= b[k];
            acc_c ^= c[k];
            acc_d ^= d[k];
        }
    }

    unsigned long long t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    unsigned int acc = acc_a ^ acc_b ^ acc_c ^ acc_d;
    unsigned int tid_g = blockIdx.x * blockDim.x + threadIdx.x;
    ((unsigned int*)C)[tid_g * 4 + 0] = acc;
    ((unsigned int*)C)[tid_g * 4 + 1] = (unsigned int)(t1 - t0);
    ((unsigned int*)C)[tid_g * 4 + 2] = (unsigned int)((t1 - t0) >> 32);
    ((unsigned int*)C)[tid_g * 4 + 3] = N_ILP * N_LOADS_OUTER;
}
