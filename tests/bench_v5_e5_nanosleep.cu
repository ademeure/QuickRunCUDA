// V5 E5: PTX nanosleep — sleep accuracy + power impact
// Test sleep durations: 0, 100ns, 1us, 10us, 100us, 1ms
#ifndef NS
#define NS 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int sleep_ns = (unsigned)NS;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // PTX nanosleep with N ns argument
        asm volatile("nanosleep.u32 %0;" :: "r"(sleep_ns));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (sleep_ns == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = sleep_ns;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // ITERS sleeps × NS ns expected; measure actual cy
        double cy_per_sleep = (double)(t1-t0)/(double)ITERS;
        double ns_per_sleep = cy_per_sleep / 1.5;  // 1500 MHz = 1.5 cy/ns
        printf("nanosleep(%dns): clk=%llu cy/sleep=%.2f ns_actual=%.2f (req=%dns ratio=%.2fx)\n",
               (int)sleep_ns, t1-t0, cy_per_sleep, ns_per_sleep, NS,
               (NS > 0) ? ns_per_sleep / (double)NS : 0.0);
    }
}
