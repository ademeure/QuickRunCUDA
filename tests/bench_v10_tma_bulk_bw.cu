// V10: cp.async.bulk (TMA non-tensor) load BW with mbarrier completion
// Use mbarrier::complete_tx::bytes for completion mechanism
#ifndef BYTES
#define BYTES 128
#endif
#ifndef K_INNER
#define K_INNER 8
#endif

extern "C" __global__ __launch_bounds__(128, 1)
void kernel(float4* A, float4* B, float4* C, int ITERS, int seed, int u2) {
    __shared__ alignas(128) unsigned long long bar;
    __shared__ alignas(16) unsigned char smem_data[BYTES * K_INNER];

    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    if (tid == 0) {
        unsigned int bar_addr = (unsigned)__cvta_generic_to_shared(&bar);
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(bar_addr));
    }
    __syncthreads();

    unsigned int bar_addr = (unsigned)__cvta_generic_to_shared(&bar);
    unsigned int total_bytes = BYTES * K_INNER;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        if (tid == 0) {
            // Set expected tx bytes BEFORE issuing copies
            unsigned long long state;
            asm volatile(
                "mbarrier.arrive.expect_tx.shared.b64 %0, [%1], %2;\n"
                : "=l"(state) : "r"(bar_addr), "r"(total_bytes)
            );
            // Issue K_INNER bulk loads
            #pragma unroll
            for (int k = 0; k < K_INNER; k++) {
                unsigned int dst = (unsigned)__cvta_generic_to_shared(&smem_data[k * BYTES]);
                // Each iter visits a fresh region of A
                unsigned long long offset = (unsigned long long)i * 37888 + (unsigned long long)gtid + k * 8;
                unsigned long long src = (unsigned long long)(A + offset);
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];\n"
                    :: "r"(dst), "l"(src), "n"(BYTES), "r"(bar_addr)
                );
            }
            // Wait for completion via test_wait spin
            asm volatile(
                "{\n"
                ".reg .pred P;\n"
                "L_wait_%=:\n"
                "mbarrier.test_wait.shared.b64 P, [%0], %1;\n"
                "@!P bra L_wait_%=;\n"
                "}\n"
                :: "r"(bar_addr), "l"(state)
            );
            // Re-init barrier for next iter
            asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(bar_addr));
        }
        __syncthreads();
    }

    // Anti-DCE
    unsigned char v = smem_data[tid & (BYTES - 1)];
    if ((int)v == seed) C[blockIdx.x * blockDim.x + tid] = make_float4((float)v, 0, 0, 0);
}
