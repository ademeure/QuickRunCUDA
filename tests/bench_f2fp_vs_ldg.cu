// F2FP contention with memory LDG/STG.
// Mode 0: LDG (from large L2-resident buffer)
// Mode 1: STG
// Mode 2: LDG.U + F2FP (LDG from uniform)
// Mode 3: LDS (shared memory)
// Mode 4: tensor-core HMMA (f16 x f16 + f32)

#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef N_CVT
#define N_CVT 16
#endif
#ifndef N_MEM
#define N_MEM 0
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MEM_KIND
#define MEM_KIND 0
#endif

#if MEM_KIND == 3
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    __shared__ unsigned int smem[4096];
    if (threadIdx.x < 4096) smem[threadIdx.x] = threadIdx.x * 0x12345u;
    __syncthreads();

    unsigned int p[N_CVT];
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);
    unsigned int macc[N_MEM > 0 ? N_MEM : 1];
    #pragma unroll
    for (int m = 0; m < N_MEM; m++) macc[m] = 0;

    unsigned int idx0 = threadIdx.x & 1023;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CVT; k++) {
                asm volatile(
                    "{ .reg .b16 _h;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h, %0;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 %0, _h; }"
                    : "+r"(p[k]));
            }
            #pragma unroll
            for (int m = 0; m < N_MEM; m++) {
                unsigned int v;
                unsigned int idx = (idx0 + m*7 + j*3) & 1023;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "l"((size_t)(smem + idx)));
                macc[m] ^= v;
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) acc ^= p[k];
    #pragma unroll
    for (int m = 0; m < N_MEM; m++) acc ^= macc[m];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
#else
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    unsigned int p[N_CVT];
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);
    unsigned int macc[N_MEM > 0 ? N_MEM : 1];
    #pragma unroll
    for (int m = 0; m < N_MEM; m++) macc[m] = 0;

    // Base pointers into A (for LDG) and C (for STG).
    // Use a small working set (8 KB per SM block) to stay in L1/L2.
    size_t off = (threadIdx.x & 255) * 4;
    unsigned int* Au = (unsigned int*)A + off;
    unsigned int* Cu = (unsigned int*)C + off;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CVT; k++) {
                asm volatile(
                    "{ .reg .b16 _h;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h, %0;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 %0, _h; }"
                    : "+r"(p[k]));
            }
            #pragma unroll
            for (int m = 0; m < N_MEM; m++) {
#if MEM_KIND == 0
                // LDG (global) — using volatile to prevent folding
                unsigned int v;
                asm volatile("ld.global.ca.b32 %0, [%1];" : "=r"(v) : "l"(Au + m) : "memory");
                macc[m] ^= v;
#elif MEM_KIND == 1
                // STG global
                asm volatile("st.global.b32 [%0], %1;" :: "l"(Cu + m), "r"(macc[m] + j) : "memory");
#endif
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) acc ^= p[k];
    #pragma unroll
    for (int m = 0; m < N_MEM; m++) acc ^= macc[m];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
#endif
