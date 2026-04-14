// HMMA solo throughput
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef N_MMA
#define N_MMA 4
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    float d[4*N_MMA];
    #pragma unroll
    for (int k = 0; k < 4*N_MMA; k++) d[k] = (threadIdx.x + k) * 0.01f;
    unsigned int a0 = 0x3C003C00u ^ threadIdx.x;
    unsigned int a1 = 0x3C003C01u ^ threadIdx.x;
    unsigned int a2 = 0x3C003C02u ^ threadIdx.x;
    unsigned int a3 = 0x3C003C03u ^ threadIdx.x;
    unsigned int b0 = 0x3C003C00u ^ (threadIdx.x + 7);
    unsigned int b1 = 0x3C003C01u ^ (threadIdx.x + 7);
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int m = 0; m < N_MMA; m++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n\t"
                    : "+f"(d[m*4+0]), "+f"(d[m*4+1]), "+f"(d[m*4+2]), "+f"(d[m*4+3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < 4*N_MMA; k++) acc ^= __float_as_int(d[k]);
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
