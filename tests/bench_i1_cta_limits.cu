// I1: Active CTA limit per SM via cudaOccupancyMaxActiveBlocksPerMultiprocessor
// We use a kernel that varies SMEM and reg pressure; query is via API
// Here we MEASURE by launching N blocks and checking how many run concurrently
// Using clock64-based timestamping
#ifndef SMEMSZ
#define SMEMSZ 0
#endif
#ifndef NREGS
#define NREGS 64
#endif
#ifndef BSZ
#define BSZ 256
#endif

#if NREGS == 32
#define LB __launch_bounds__(BSZ, 8)
#elif NREGS == 64
#define LB __launch_bounds__(BSZ, 4)
#elif NREGS == 128
#define LB __launch_bounds__(BSZ, 2)
#elif NREGS == 256
#define LB __launch_bounds__(BSZ, 1)
#endif

extern "C" __global__ LB
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#if SMEMSZ > 0
    extern __shared__ unsigned int smem[];

    // Touch SMEM to ensure it's allocated
    if (threadIdx.x == 0) {
        smem[0] = 1;
        smem[SMEMSZ/4 - 1] = 2;
    }
#endif

    // Each block records start clock; then long sleep; then end clock
    __syncthreads();

    unsigned long long t_start, t_end;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_start));

    // Loop with FFMA to consume time + use NREGS registers
    float accs[8];
    for (int k = 0; k < 8; k++) accs[k] = (float)threadIdx.x + (float)u2 + (float)k;
    float ya = (float)threadIdx.x + 0.001f, za = 0.5f;

    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) accs[k] = accs[k] * ya + za;
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_end));

    // Each block writes its [start, end] to C[bid*2..2+1]
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[blockIdx.x * 2 + 0] = t_start;
        ((unsigned long long*)C)[blockIdx.x * 2 + 1] = t_end;
#if SMEMSZ > 0
        ((unsigned int*)C)[blockIdx.x + 4096] = (unsigned int)smem[0]; // sentinel
#endif
    }
    if (threadIdx.x == 0 && accs[0] == 12345.6f) C[8192 + blockIdx.x] = accs[0];
}
