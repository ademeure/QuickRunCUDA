// 32x32 SHMEM transpose: bank-conflict comparison.
// One warp loads 32 floats per thread (32×32 tile), transposes via SHMEM, writes back.
//
// MODE 0: Naive — SMEM[i][j] -> SMEM[j][i] -> 32-way bank conflict on read
// MODE 1: Padded — SMEM[i][j+pad] -> conflict-free
// MODE 2: Skewed — SMEM[i][(j+i)&31] -> conflict-free via skew, no padding
// MODE 3: stmatrix/ldmatrix (m8n8) — tensor-tile transpose intrinsics

#ifndef MODE
#define MODE 0
#endif
#ifndef N_ITERS
#define N_ITERS 10000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

#if MODE == 1
    __shared__ float smem[32][33];  // padded
#else
    __shared__ float smem[32][32];
#endif

    // Each lane has 32 values to write/read
    float vals[32];
    #pragma unroll
    for (int i = 0; i < 32; i++)
        vals[i] = (float)(threadIdx.x * 32 + i + (unsigned)u2);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
#if MODE == 0
        // Naive: lane k writes column k (smem[i][k]), reads row k (smem[k][i])
        #pragma unroll
        for (int i = 0; i < 32; i++) smem[i][threadIdx.x] = vals[i];
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 32; i++) vals[i] = smem[threadIdx.x][i];
        __syncwarp();
#elif MODE == 1
        // Padded: same writes/reads but smem has +1 column
        #pragma unroll
        for (int i = 0; i < 32; i++) smem[i][threadIdx.x] = vals[i];
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 32; i++) vals[i] = smem[threadIdx.x][i];
        __syncwarp();
#elif MODE == 2
        // Skewed: lane k writes smem[i][(k+i)&31]
        #pragma unroll
        for (int i = 0; i < 32; i++) smem[i][(threadIdx.x + i) & 31] = vals[i];
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 32; i++) vals[i] = smem[threadIdx.x][(i + threadIdx.x) & 31];
        __syncwarp();
#elif MODE == 3
        // stmatrix/ldmatrix-like: just write+read via smem column-major as if it were
        // already transposed (no actual transpose, used as throughput baseline).
        #pragma unroll
        for (int i = 0; i < 32; i++) smem[threadIdx.x][i] = vals[i];
        __syncwarp();
        #pragma unroll
        for (int i = 0; i < 32; i++) vals[i] = smem[threadIdx.x][i];
        __syncwarp();
#endif
        // Defeat hoisting — perturb vals
        vals[0] += (float)((unsigned)u2 * (unsigned)it);
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) sum += vals[i];
    if ((int)sum == seed) C[blockIdx.x] = sum;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_ITERS=%d clk=%llu cy/iter=%.3f\n",
               MODE, N_ITERS, t1 - t0, (double)(t1-t0)/(double)N_ITERS);
    }
}
