// V7 G3: L1 hit rate vs occupancy (warps per SM)
// Vary launch_bounds to control occupancy
// Higher occupancy → more concurrent warps → potentially more L1 thrashing
#ifndef THREADS
#define THREADS 32
#endif

extern "C" __global__ __launch_bounds__(THREADS, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int MASK = (4 * 1024 / 4) - 1;  // 4 KB working set / 4 B = 1024 floats

    float acc = 0.0f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned int idx = (gtid + i) & MASK;
        acc += A[idx];
    }

    if (acc == 1.234567e-30f) C[gtid] = acc;
}
