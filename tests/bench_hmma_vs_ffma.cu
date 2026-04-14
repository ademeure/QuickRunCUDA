// HMMA vs FFMA co-issue — tests whether HMMA also displaces FFMA issue slots
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef N_FMA
#define N_FMA 8
#endif
#ifndef N_MMA
#define N_MMA 0
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    float r[N_FMA];
    #pragma unroll
    for (int k = 0; k < N_FMA; k++) r[k] = __int_as_float(threadIdx.x | 0x3F800000u) + k;
    float d[4*(N_MMA>0?N_MMA:1)];
    #pragma unroll
    for (int k = 0; k < 4*N_MMA; k++) d[k] = (threadIdx.x + k) * 0.01f;
    unsigned int a0 = 0x3C003C00u ^ threadIdx.x;
    unsigned int a1 = 0x3C003C01u ^ threadIdx.x;
    unsigned int a2 = 0x3C003C02u ^ threadIdx.x;
    unsigned int a3 = 0x3C003C03u ^ threadIdx.x;
    unsigned int b0 = 0x3C003C00u ^ (threadIdx.x + 7);
    unsigned int b1 = 0x3C003C01u ^ (threadIdx.x + 7);
    const float ca = 1.0000001f;
    const float cb = 0.9999999f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_FMA; k++) {
                asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(r[k]) : "f"(ca), "f"(cb));
            }
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
    for (int k = 0; k < N_FMA; k++) acc ^= __float_as_int(r[k]);
    #pragma unroll
    for (int k = 0; k < 4*N_MMA; k++) acc ^= __float_as_int(d[k]);
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
