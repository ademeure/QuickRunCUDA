// V5 G6: Top-K (K=4) over 1024 floats SoL
// Strategy: per-thread keep K=4 best values; warp/block reduce by repeated argmax
// Simpler: each thread tracks K=4 best; merge via SHFL-based selection
//
// Approach: 256 threads × 4 elements = 1024 total.
// Each thread already has its 4 values; just need to select top 4 across all.
// Use 4 rounds of argmax with the chosen value masked out (set to -inf)
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(16) float smem_v[1024];
    __shared__ unsigned topk_idx[4];
    __shared__ float topk_v[4];
    __shared__ float final_v;
    __shared__ unsigned final_idx;

    int tid = threadIdx.x;
    smem_v[tid] = (float)((tid * 13 + (unsigned)u2) & 0x3FF);
    smem_v[tid + 256] = (float)(((tid + 256) * 13 + (unsigned)u2) & 0x3FF);
    smem_v[tid + 512] = (float)(((tid + 512) * 13 + (unsigned)u2) & 0x3FF);
    smem_v[tid + 768] = (float)(((tid + 768) * 13 + (unsigned)u2) & 0x3FF);
    __syncthreads();

    unsigned long long t0, t1;
    if (tid == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < ITERS; it++) {
        // Each thread holds 4 elements
        float v[4] = {smem_v[tid], smem_v[tid+256], smem_v[tid+512], smem_v[tid+768]};
        unsigned idx[4] = {(unsigned)tid, (unsigned)tid+256, (unsigned)tid+512, (unsigned)tid+768};

        // K=4 rounds of argmax
        for (int k = 0; k < 4; k++) {
            // Per-thread: find local argmax over its 4 (with already-chosen masked)
            float best_v = -1e30f;
            unsigned best_idx = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (v[j] > best_v) { best_v = v[j]; best_idx = idx[j]; }
            }

            // Warp reduce
            for (int s = 16; s > 0; s >>= 1) {
                float other_v = __shfl_xor_sync(0xFFFFFFFF, best_v, s);
                unsigned other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, s);
                if (other_v > best_v) { best_v = other_v; best_idx = other_idx; }
            }

            // Cross-warp via SMEM
            __shared__ float warp_v[8];
            __shared__ unsigned warp_idx[8];
            if ((tid & 31) == 0) {
                warp_v[tid >> 5] = best_v;
                warp_idx[tid >> 5] = best_idx;
            }
            __syncthreads();
            if (tid < 32) {
                float m_v = (tid < 8) ? warp_v[tid] : -1e30f;
                unsigned m_idx = (tid < 8) ? warp_idx[tid] : 0;
                for (int s = 4; s > 0; s >>= 1) {
                    float other_v = __shfl_xor_sync(0xFFFFFFFF, m_v, s);
                    unsigned other_idx = __shfl_xor_sync(0xFFFFFFFF, m_idx, s);
                    if (other_v > m_v) { m_v = other_v; m_idx = other_idx; }
                }
                if (tid == 0) {
                    topk_v[k] = m_v;
                    topk_idx[k] = m_idx;
                    final_v = m_v;
                    final_idx = m_idx;
                }
            }
            __syncthreads();

            // Mask out the chosen value in this thread's 4 (if it was ours)
            unsigned chosen = topk_idx[k];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (idx[j] == chosen) v[j] = -1e30f;
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        printf("Top-4 1024: clk=%llu cy/topk=%.0f time_ns=%.0f (top: %u %u %u %u)\n",
               t1-t0, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS / 1.5,
               topk_idx[0], topk_idx[1], topk_idx[2], topk_idx[3]);
    }
    if (final_idx == (unsigned)seed) C[blockIdx.x] = (float)final_idx;
}
