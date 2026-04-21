// V8 L4: per-warp instruction trace
// Each warp timestamps entry/exit to measure scheduling variance.
// Test at full occupancy (148 SMs × 2048 threads = 9472 warps total)
// and compare variance vs expected from fair scheduler.

#ifndef WORK_CYCLES
#define WORK_CYCLES 1024
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(unsigned long long* trace, float* B, float* C, int ITERS, int seed, int u2) {
    int warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    unsigned long long t_start = 0, t_end = 0;
    if (lane == 0) {
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t_start));
    }

    // Fixed work: FFMA loop with distinct sources to defeat DCE
    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(threadIdx.x + 2) * 0.001f;
    float c = a;
    #pragma unroll 32
    for (int i = 0; i < WORK_CYCLES; i++) {
        c = a * c + b;
    }

    if (lane == 0) {
        asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t_end));
        trace[warp_id * 2 + 0] = t_start;
        trace[warp_id * 2 + 1] = t_end;
    }

    // Anti-DCE
    if (c == 1.234567e-30f) ((float*)trace)[warp_id * 4] = c;
}

extern "C" __global__ void init(unsigned long long* trace, float* B, float* C, int ITERS, int seed, int u2) {
    int warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (lane == 0) {
        trace[warp_id * 2 + 0] = 0;
        trace[warp_id * 2 + 1] = 0;
    }
}
