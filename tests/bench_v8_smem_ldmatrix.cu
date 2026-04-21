// V8 SMEM via ldmatrix — MEASURE SoL with proper DCE defeat
// ldmatrix.sync.aligned.m8n8.x4.shared.b16 delivers 4 × 8×8 × 2B = 512 B per warp issue.
// Theoretical: 148 SMs × 512 B / cy × 4 SMSPs × 1.92 GHz = way-too-high
// Actual cap is bank throughput: 32 × 4 B × 4 bank-sets = 512 B/cy/SM at best
//   148 × 512 × 1.92e9 / 4 = 37 TB/s (if ldmatrix fully uses 4 SMSPs concurrently)
//
// DCE fix: accumulate results into independent sum registers per iter; sum written to global.

#ifndef SMEM_WORDS
#define SMEM_WORDS 8192   // 32 KB per block
#endif

extern "C" __global__ __launch_bounds__(128, 8)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ alignas(128) unsigned int smem[SMEM_WORDS];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    // Init SMEM
    #pragma unroll
    for (int i = tid; i < SMEM_WORDS; i += blockDim.x) {
        smem[i] = (unsigned int)A[(gtid + i) & (SMEM_WORDS - 1)] ^ (unsigned int)i;
    }
    __syncthreads();

    int lane = tid & 31;
    int warp_id = tid >> 5;

    // For x1: 1 matrix (8×8 halves = 128 B per warp per issue)
    // Lanes 0-7 provide row pointers; lanes 8-31 unused for address but all get a result half
    // Stride: rows of 16 B each → addresses at lane*4 (4 words = 16 B) for lanes 0-7
    // To avoid conflicts: put each warp's matrix in its own SMEM slice
    unsigned int sum = 0;
    int warp_base = (warp_id * 64) & (SMEM_WORDS - 1);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Each warp uses a disjoint 128-B region. Lanes 0-7 provide row pointers (stride 4 words).
        int iter_off = (i * 16) & (SMEM_WORDS - 1);   // move 64 B per iter
        int local = (warp_base + (lane & 7) * 4 + iter_off) & (SMEM_WORDS - 1);
        unsigned int ptr = __cvta_generic_to_shared(&smem[local]);

        unsigned int r0;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
            : "=r"(r0)
            : "r"(ptr)
        );
        sum += r0;
    }

    unsigned int total = sum;
    if (total == 0xFFFFFFFF) C[gtid] = __int_as_float((int)total);
}
