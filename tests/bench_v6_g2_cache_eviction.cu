// V6 G2: L1 vs L2 cache eviction — measure miss patterns at different working set sizes
// L1 capacity: ~192 KB per SM (4096 cache lines × 48 bytes? need check)
// L2 capacity: 60 MB shared
// Sweep: read N float4 entries, varying N, measure cy/access
// Stride pattern with WORKING_SET_KB controlling reuse distance
#ifndef WORKING_SET_KB
#define WORKING_SET_KB 64
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // working set in bytes / 16 bytes per float4 / total threads = elements per thread
    unsigned int n_elems = (WORKING_SET_KB * 1024) / 16;
    unsigned int mask = n_elems - 1;  // assume power of 2

    float4 sum = make_float4(0,0,0,0);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned int idx = (gtid + i) & mask;
        float4 v = A[idx];
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (sum.x == 1.234567e-30f) C[blockIdx.x * blockDim.x + threadIdx.x] = sum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("WORKING_SET_KB=%d ITERS=%d cy/access=%.3f\n",
               WORKING_SET_KB, ITERS, (double)(t1-t0)/(double)ITERS);
    }
}
