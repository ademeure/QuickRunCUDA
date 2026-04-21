// V5 G5: Argmax 1024 floats — return idx of max value
// Each thread takes 4 elements; argmax-reduce via SHFL
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(16) float smem_v[1024];
    __shared__ __align__(16) unsigned smem_idx[1024];
    __shared__ float final_v;
    __shared__ unsigned final_idx;

    int tid = threadIdx.x;
    smem_v[tid] = (float)((tid * 13 + (unsigned)u2) & 0xFF);
    smem_v[tid + 256] = (float)(((tid + 256) * 13 + (unsigned)u2) & 0xFF);
    smem_v[tid + 512] = (float)(((tid + 512) * 13 + (unsigned)u2) & 0xFF);
    smem_v[tid + 768] = (float)(((tid + 768) * 13 + (unsigned)u2) & 0xFF);
    __syncthreads();

    unsigned long long t0, t1;
    if (tid == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < ITERS; it++) {
        // Per-thread: 4 elements, find local argmax
        float v0 = smem_v[tid];
        float v1 = smem_v[tid + 256];
        float v2 = smem_v[tid + 512];
        float v3 = smem_v[tid + 768];
        float best_v = v0;
        unsigned best_idx = tid;
        if (v1 > best_v) { best_v = v1; best_idx = tid + 256; }
        if (v2 > best_v) { best_v = v2; best_idx = tid + 512; }
        if (v3 > best_v) { best_v = v3; best_idx = tid + 768; }

        // Warp reduce via __shfl_xor: keep value + index together
        for (int s = 16; s > 0; s >>= 1) {
            float other_v = __shfl_xor_sync(0xFFFFFFFF, best_v, s);
            unsigned other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, s);
            if (other_v > best_v) { best_v = other_v; best_idx = other_idx; }
        }
        // Lane 0 of each warp has warp argmax

        // Cross-warp via SMEM
        __shared__ float warp_v[8];
        __shared__ unsigned warp_idx[8];
        if ((tid & 31) == 0) {
            warp_v[tid >> 5] = best_v;
            warp_idx[tid >> 5] = best_idx;
        }
        __syncthreads();

        // Single warp 0 reduces 8 → 1
        if (tid < 32) {
            float v = (tid < 8) ? warp_v[tid] : -1.0e30f;
            unsigned idx = (tid < 8) ? warp_idx[tid] : 0;
            for (int s = 4; s > 0; s >>= 1) {
                float other_v = __shfl_xor_sync(0xFFFFFFFF, v, s);
                unsigned other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, s);
                if (other_v > v) { v = other_v; idx = other_idx; }
            }
            if (tid == 0) {
                final_v = v;
                final_idx = idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        printf("Argmax 1024 elem: clk=%llu cy/argmax=%.0f time_ns=%.0f (idx=%u v=%.0f)\n",
               t1-t0, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS / 1.5,
               final_idx, final_v);
    }
    if (final_idx == (unsigned)seed) C[blockIdx.x] = (float)final_idx;
}
