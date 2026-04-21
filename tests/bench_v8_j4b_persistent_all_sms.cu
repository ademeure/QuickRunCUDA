// V8 J4b: Power comparison — persistent kernel variants
// MODE 0: 1 SM × 1 thread spin (V7 J4 original)
// MODE 1: 148 SMs × 1 thread spin
// MODE 2: 148 SMs × 256 threads spin (full occupancy)
// MODE 3: 148 SMs × 256 threads __nanosleep(1000) loop
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    volatile unsigned int* sig = (volatile unsigned int*)A;

#if MODE == 0
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < ITERS; i++) while (*sig != (unsigned)i) {}
#elif MODE == 1
    if (threadIdx.x != 0) return;
    for (int i = 0; i < ITERS; i++) while (*sig != (unsigned)i) {}
#elif MODE == 2
    // All threads spin
    for (int i = 0; i < ITERS; i++) while (*sig != (unsigned)i) {}
#elif MODE == 3
    // All threads nanosleep
    for (int i = 0; i < ITERS; i++) {
        while (*sig != (unsigned)i) {
            __nanosleep(1000);
        }
    }
#endif
}
