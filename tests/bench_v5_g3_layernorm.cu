// V5 G3: LayerNorm 1024 elements SoL
// 2-pass algo:
//   1. mean = sum(x) / N
//   2. variance = sum((x - mean)^2) / N
//   3. y[i] = (x[i] - mean) * rsqrt(var + eps) * gain[i] + bias[i]
// Welford-style would be 1-pass but more complex; use 2-pass for clarity
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(16) float smem[1024];
    __shared__ __align__(16) float gain[1024];
    __shared__ __align__(16) float bias[1024];
    __shared__ float mean_shared;
    __shared__ float invstd_shared;

    int tid = threadIdx.x;
    smem[tid]       = (float)(tid + (unsigned)u2) * 0.001f;
    smem[tid + 256] = (float)(tid + 256 + (unsigned)u2) * 0.001f;
    smem[tid + 512] = (float)(tid + 512 + (unsigned)u2) * 0.001f;
    smem[tid + 768] = (float)(tid + 768 + (unsigned)u2) * 0.001f;
    gain[tid] = 1.0f; gain[tid+256] = 1.0f; gain[tid+512] = 1.0f; gain[tid+768] = 1.0f;
    bias[tid] = 0.0f; bias[tid+256] = 0.0f; bias[tid+512] = 0.0f; bias[tid+768] = 0.0f;
    __syncthreads();

    unsigned long long t0, t1;
    if (tid == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < ITERS; it++) {
        float v0 = smem[tid];
        float v1 = smem[tid + 256];
        float v2 = smem[tid + 512];
        float v3 = smem[tid + 768];

        // Pass 1: sum
        float s = v0 + v1 + v2 + v3;
        for (int st = 16; st > 0; st >>= 1) s += __shfl_xor_sync(0xFFFFFFFF, s, st);
        __shared__ float warp_s[8];
        if ((tid & 31) == 0) warp_s[tid >> 5] = s;
        __syncthreads();
        if (tid < 32) {
            float m = (tid < 8) ? warp_s[tid] : 0;
            for (int st = 4; st > 0; st >>= 1) m += __shfl_xor_sync(0xFFFFFFFF, m, st);
            if (tid == 0) mean_shared = m * (1.0f / 1024.0f);
        }
        __syncthreads();
        float mean = mean_shared;

        // Pass 2: variance
        float d0 = v0 - mean, d1 = v1 - mean, d2 = v2 - mean, d3 = v3 - mean;
        float var = d0*d0 + d1*d1 + d2*d2 + d3*d3;
        for (int st = 16; st > 0; st >>= 1) var += __shfl_xor_sync(0xFFFFFFFF, var, st);
        __shared__ float warp_var[8];
        if ((tid & 31) == 0) warp_var[tid >> 5] = var;
        __syncthreads();
        if (tid < 32) {
            float vv = (tid < 8) ? warp_var[tid] : 0;
            for (int st = 4; st > 0; st >>= 1) vv += __shfl_xor_sync(0xFFFFFFFF, vv, st);
            if (tid == 0) {
                float invstd;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(invstd) : "f"(vv * (1.0f/1024.0f) + 1e-5f));
                invstd_shared = invstd;
            }
        }
        __syncthreads();
        float invstd = invstd_shared;

        // Pass 3: normalize + affine
        smem[tid]       = d0 * invstd * gain[tid]       + bias[tid];
        smem[tid + 256] = d1 * invstd * gain[tid + 256] + bias[tid + 256];
        smem[tid + 512] = d2 * invstd * gain[tid + 512] + bias[tid + 512];
        smem[tid + 768] = d3 * invstd * gain[tid + 768] + bias[tid + 768];
        __syncthreads();
    }

    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        printf("LayerNorm 1024 elem: clk=%llu cy/ln=%.0f time_ns=%.0f\n",
               t1-t0, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS / 1.5);
    }
    if (smem[1023] == 0xDEADBEEF) C[blockIdx.x] = smem[1023];
}
