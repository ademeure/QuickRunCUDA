// V6 G6: Constant cache vs L1 cache vs Global memory
// MODE 0: __constant__ array (uses constant cache)
// MODE 1: __shared__ array (SMEM)
// MODE 2: Global mem with hot reads (L1 cached)
//
// Use 32 threads × 8 chained reads from same address (broadcast pattern, ideal for cmem)
#ifndef MODE
#define MODE 0
#endif

__constant__ float cmem_data[1024];

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ float smem_data[1024];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 1024; i++) smem_data[i] = (float)i;
    }
    __syncwarp();

    float acc = 0.0f;
    int idx = threadIdx.x;  // each thread reads same index for broadcast

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // 8 constant reads (broadcast)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v = cmem_data[(i + k) & 1023];
            acc += v;
        }
#elif MODE == 1
        // 8 SMEM reads (broadcast)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v;
            unsigned int sa = (unsigned int)__cvta_generic_to_shared(&smem_data[(i + k) & 1023]);
            asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v) : "r"(sa));
            acc += v;
        }
#elif MODE == 2
        // 8 global reads (broadcast — same addr across warp)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float v = A[(i + k) & 1023];
            acc += v;
        }
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == 1.234567e-30f) C[blockIdx.x] = acc;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f cy/op=%.3f\n",
               MODE, ITERS, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/8.0);
    }
}
