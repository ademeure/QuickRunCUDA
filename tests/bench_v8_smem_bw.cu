// V8 SMEM-BW: measure B300 shared memory bandwidth ceiling
// Theoretical: 32 banks × 4 B × 2.032 GHz × 148 SMs = 38.5 TB/s at boost.
// Pattern: each thread loads `ITERS` floats from SMEM, each from a distinct bank.
// Bank stride = 4 bytes per thread so warp is conflict-free.
//
// Anti-DCE: sum write to C[] under impossible condition.

#ifndef SMEM_SIZE
#define SMEM_SIZE 8192    // 32 KB/block (fits in 228 KB SMEM)
#endif

extern "C" __global__ __launch_bounds__(128, 8)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ float smem[SMEM_SIZE];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    // Initialize SMEM (cooperative)
    #pragma unroll
    for (int i = tid; i < SMEM_SIZE; i += blockDim.x) {
        smem[i] = A[(gtid + i) & (SMEM_SIZE - 1)];
    }
    __syncthreads();

    // Sustained SMEM reads — heavy unroll for dispatch-rate
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    float a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    int base = tid;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        int idx = (base + i * 128) & (SMEM_SIZE - 1);
        // 8-way ILP + unroll 16 inner → 128 loads per outer iter
        #pragma unroll 16
        for (int j = 0; j < 16; j++) {
            a0 += smem[(idx + j * 128 + 0)    & (SMEM_SIZE - 1)];
            a1 += smem[(idx + j * 128 + 16)   & (SMEM_SIZE - 1)];
            a2 += smem[(idx + j * 128 + 32)   & (SMEM_SIZE - 1)];
            a3 += smem[(idx + j * 128 + 48)   & (SMEM_SIZE - 1)];
            a4 += smem[(idx + j * 128 + 64)   & (SMEM_SIZE - 1)];
            a5 += smem[(idx + j * 128 + 80)   & (SMEM_SIZE - 1)];
            a6 += smem[(idx + j * 128 + 96)   & (SMEM_SIZE - 1)];
            a7 += smem[(idx + j * 128 + 112)  & (SMEM_SIZE - 1)];
        }
    }

    float sum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (sum == 1.234567e-30f) C[gtid] = sum;
}
