// __threadfence variants cost
// Mode 0: no fence (baseline)
// Mode 1: __threadfence_block
// Mode 2: __threadfence (GPU)
// Mode 3: __threadfence_system
// Mode 4: PTX membar.cta
// Mode 5: PTX membar.gl
// Mode 6: PTX membar.sys

#ifndef MODE
#define MODE 0
#endif
#ifndef N_ITERS
#define N_ITERS 100000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)threadIdx.x;
    int* p = (int*)A;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_ITERS; i++) {
        v = v * 31u + (unsigned)i;
#if MODE == 1
        __threadfence_block();
#elif MODE == 2
        __threadfence();
#elif MODE == 3
        __threadfence_system();
#elif MODE == 4
        asm volatile("membar.cta;");
#elif MODE == 5
        asm volatile("membar.gl;");
#elif MODE == 6
        asm volatile("membar.sys;");
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_ITERS=%d clk=%llu cy/iter=%.3f\n",
               MODE, N_ITERS, t1 - t0, (double)(t1-t0)/(double)N_ITERS);
    }
}
