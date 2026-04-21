// V5 G4: RMSNorm SoL — 1024 floats, single block
// Algorithm:
//   sumsq = sum(x[i]^2)
//   scale = rsqrt(sumsq / 1024 + eps)
//   y[i] = x[i] * gain[i] * scale
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(16) float smem[1024];
    __shared__ __align__(16) float gain[1024];
    __shared__ float scale_shared;

    int tid = threadIdx.x;

    // Init x and gain (per iter; defeat folding)
    smem[tid] = (float)(tid + (unsigned)u2) * 0.001f;
    smem[tid + 256] = (float)(tid + 256 + (unsigned)u2) * 0.001f;
    smem[tid + 512] = (float)(tid + 512 + (unsigned)u2) * 0.001f;
    smem[tid + 768] = (float)(tid + 768 + (unsigned)u2) * 0.001f;

    gain[tid] = 1.0f + (float)tid * 0.0001f;
    gain[tid + 256] = 1.0f;
    gain[tid + 512] = 1.0f;
    gain[tid + 768] = 1.0f;
    __syncthreads();

    unsigned long long t0, t1;
    if (tid == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < ITERS; it++) {
        // Step 1: each thread reads 4 elements, computes sum of squares
        float v0 = smem[tid];
        float v1 = smem[tid + 256];
        float v2 = smem[tid + 512];
        float v3 = smem[tid + 768];
        float sumsq = v0*v0 + v1*v1 + v2*v2 + v3*v3;

        // Warp-level reduction via redux.sync.add (FAST - per Q2 finding)
        // FFMA is float; need int-based redux. Convert via fp ops:
        // Actually use SHFL chain for float reduction (no float redux on SM10)
        sumsq += __shfl_xor_sync(0xFFFFFFFF, sumsq, 16);
        sumsq += __shfl_xor_sync(0xFFFFFFFF, sumsq, 8);
        sumsq += __shfl_xor_sync(0xFFFFFFFF, sumsq, 4);
        sumsq += __shfl_xor_sync(0xFFFFFFFF, sumsq, 2);
        sumsq += __shfl_xor_sync(0xFFFFFFFF, sumsq, 1);
        // Warp 0 has full warp sum

        // Cross-warp via SMEM
        __shared__ float warp_sums[8];
        if ((tid & 31) == 0) warp_sums[tid >> 5] = sumsq;
        __syncthreads();

        // Single warp reduces 8 warp sums
        float total = 0;
        if (tid < 32) {
            total = (tid < 8) ? warp_sums[tid] : 0;
            total += __shfl_xor_sync(0xFFFFFFFF, total, 4);
            total += __shfl_xor_sync(0xFFFFFFFF, total, 2);
            total += __shfl_xor_sync(0xFFFFFFFF, total, 1);
            if (tid == 0) {
                // total = sum of all squares; mean = total / 1024
                float scale = rsqrtf(total / 1024.0f + 1e-6f);
                scale_shared = scale;
            }
        }
        __syncthreads();
        float scale = scale_shared;

        // Step 3: write back y[i] = x[i] * gain[i] * scale
        smem[tid] = v0 * gain[tid] * scale;
        smem[tid + 256] = v1 * gain[tid + 256] * scale;
        smem[tid + 512] = v2 * gain[tid + 512] * scale;
        smem[tid + 768] = v3 * gain[tid + 768] * scale;
        __syncthreads();
    }

    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        printf("RMSNorm 1024 elem SMEM: clk=%llu cy/norm=%.0f time_ns=%.0f\n",
               t1-t0, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS / 1.5);
    }
    if (smem[1023] == 0xDEADBEEF) C[blockIdx.x] = smem[1023];
}
