// V9: cp.async.ca.shared.global BW
// Asynchronous GMEM→SMEM load (Ampere+). Bypasses L1 register bank pressure.
// Expected: similar BW to LDG (both hit HBM), but async pipeline = no stall.

#ifndef K_INNER
#define K_INNER 64
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    __shared__ alignas(16) unsigned smem[4096];  // 16 KB
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    unsigned smem_ptr = (unsigned)__cvta_generic_to_shared(&smem[tid * 4]);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned long long base = (unsigned long long)i * 37888 * K_INNER;
        #pragma unroll 8
        for (int k = 0; k < K_INNER; k++) {
            unsigned long long idx = base + (unsigned long long)k * 37888 + gtid;
            const float4* src = A + idx;
            // cp.async.ca.shared.global.b16 — 16 B async load
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                         :: "r"(smem_ptr), "l"(src));
        }
        // Commit + wait per iter
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
    }

    __syncthreads();
    // Anti-DCE: read back from SMEM and write to C
    unsigned v = smem[tid * 4];
    if (v == 0xDEADBEEF) C[gtid] = make_float4((float)v, 0, 0, 0);
}
