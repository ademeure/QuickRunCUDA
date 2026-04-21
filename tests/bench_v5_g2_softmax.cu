// V5 G2: SoftMax 1024 elements SoL
// 3-pass algo:
//   1. max(x)
//   2. exp(x - max), sum
//   3. y = exp(x - max) / sum
// Use ex2.approx.ftz for fast exp (per .ftz finding)
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(16) float smem[1024];
    __shared__ float max_shared;
    __shared__ float sum_shared;

    int tid = threadIdx.x;
    smem[tid] = (float)((tid + (unsigned)u2) & 0xFF) * 0.01f;
    smem[tid + 256] = (float)(((tid + 256) + (unsigned)u2) & 0xFF) * 0.01f;
    smem[tid + 512] = (float)(((tid + 512) + (unsigned)u2) & 0xFF) * 0.01f;
    smem[tid + 768] = (float)(((tid + 768) + (unsigned)u2) & 0xFF) * 0.01f;
    __syncthreads();

    unsigned long long t0, t1;
    if (tid == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < ITERS; it++) {
        // Pass 1: find max
        float v0 = smem[tid];
        float v1 = smem[tid + 256];
        float v2 = smem[tid + 512];
        float v3 = smem[tid + 768];
        float local_max = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));

        // Warp reduce max via SHFL
        for (int s = 16; s > 0; s >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, s));
        }
        // Cross-warp via SMEM
        __shared__ float warp_max[8];
        if ((tid & 31) == 0) warp_max[tid >> 5] = local_max;
        __syncthreads();
        if (tid < 32) {
            float m = (tid < 8) ? warp_max[tid] : -1e30f;
            for (int s = 4; s > 0; s >>= 1) {
                m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, s));
            }
            if (tid == 0) max_shared = m;
        }
        __syncthreads();
        float global_max = max_shared;

        // Pass 2: compute exp(x - max), sum (use ex2.approx.ftz × log2(e))
        float e0, e1, e2, e3;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e0) : "f"((v0 - global_max) * 1.44269504f));
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e1) : "f"((v1 - global_max) * 1.44269504f));
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e2) : "f"((v2 - global_max) * 1.44269504f));
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e3) : "f"((v3 - global_max) * 1.44269504f));
        float local_sum = e0 + e1 + e2 + e3;

        for (int s = 16; s > 0; s >>= 1) local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, s);
        __shared__ float warp_sum[8];
        if ((tid & 31) == 0) warp_sum[tid >> 5] = local_sum;
        __syncthreads();
        if (tid < 32) {
            float s = (tid < 8) ? warp_sum[tid] : 0;
            for (int st = 4; st > 0; st >>= 1) s += __shfl_xor_sync(0xFFFFFFFF, s, st);
            if (tid == 0) sum_shared = s;
        }
        __syncthreads();
        float inv_sum;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_sum) : "f"(sum_shared));

        // Pass 3: y = exp / sum
        smem[tid] = e0 * inv_sum;
        smem[tid + 256] = e1 * inv_sum;
        smem[tid + 512] = e2 * inv_sum;
        smem[tid + 768] = e3 * inv_sum;
        __syncthreads();
    }

    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        printf("Softmax 1024 elem: clk=%llu cy/sm=%.0f time_ns=%.0f\n",
               t1-t0, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS / 1.5);
    }
    if (smem[1023] == 0xDEADBEEF) C[blockIdx.x] = smem[1023];
}
