// V8: BF16 mma.sync peak — ML training path common for gradients
// mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32
#ifndef N_CHAINS
#define N_CHAINS 8
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned a0 = (unsigned)(threadIdx.x * 17 + seed);
    unsigned a1 = (unsigned)(threadIdx.x * 23 + seed + 1);
    unsigned a2 = (unsigned)(threadIdx.x * 29 + seed + 2);
    unsigned a3 = (unsigned)(threadIdx.x * 31 + seed + 3);
    unsigned b0 = (unsigned)(threadIdx.x * 37 + seed + 4);
    unsigned b1 = (unsigned)(threadIdx.x * 41 + seed + 5);

    float c0[N_CHAINS], c1[N_CHAINS], c2[N_CHAINS], c3[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        c0[k] = (float)(threadIdx.x + k);
        c1[k] = (float)(threadIdx.x + k + 10);
        c2[k] = (float)(threadIdx.x + k + 100);
        c3[k] = (float)(threadIdx.x + k + 1000);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < N_CHAINS; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
                " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(c0[k]), "+f"(c1[k]), "+f"(c2[k]), "+f"(c3[k])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1)
            );
        }
    }

    float acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc += c0[k] + c1[k] + c2[k] + c3[k];
    if (acc == 1.234567e-30f) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
