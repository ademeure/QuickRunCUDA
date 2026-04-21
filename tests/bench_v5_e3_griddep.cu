// V5 E3: griddepcontrol PTX — programmatic dependent launch
// Used by PDL (Programmatic Dependent Launch) — kernel signals when dependent kernel can start
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned int v = (unsigned)u2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Baseline: just clock + ALU work
        v ^= i;
#elif MODE == 1
        // launch_dependents: signals dependent kernel can begin
        asm volatile("griddepcontrol.launch_dependents;");
        v ^= i;
#elif MODE == 2
        // wait: this kernel waits for parent to call launch_dependents
        asm volatile("griddepcontrol.wait;");
        v ^= i;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
