// V10: cp.async with wait_group(N) pipelining — push HBM peak
// Issue K_INNER cp.async loads, commit_group every N, wait_group(M) for overlap.
#ifndef K_INNER
#define K_INNER 64
#endif
#ifndef WAIT_DEPTH
#define WAIT_DEPTH 1   // Allow 1 group in flight (default sync). Larger = more overlap.
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    __shared__ alignas(16) unsigned smem[4096];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    unsigned smem_ptr = (unsigned)__cvta_generic_to_shared(&smem[tid * 4]);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned long long base = (unsigned long long)i * 37888 * K_INNER;
        // Issue K_INNER async loads, commit groups
        #pragma unroll 8
        for (int k = 0; k < K_INNER; k++) {
            unsigned long long idx = base + (unsigned long long)k * 37888 + gtid;
            const float4* src = A + idx;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                         :: "r"(smem_ptr), "l"(src));
        }
        asm volatile("cp.async.commit_group;");
        // Only wait for N groups to be in flight
        asm volatile("cp.async.wait_group %0;" :: "n"(WAIT_DEPTH));
    }
    // Final wait
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    unsigned v = smem[tid * 4];
    if (v == 0xDEADBEEF) C[gtid] = make_float4((float)v, 0, 0, 0);
}
