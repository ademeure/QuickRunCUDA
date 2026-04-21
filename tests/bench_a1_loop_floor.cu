// Find the empty-loop overhead in cy
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned int v = (unsigned)u2;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        v ^= i;
        asm volatile("" : "+r"(v));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("EMPTY clk=%llu cy/iter=%.3f\n", t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
