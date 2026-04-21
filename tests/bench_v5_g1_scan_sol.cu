// V5 G1: Prefix scan SoL — fix Q4 SHFL warp scan
// 1024 ints in SMEM → inclusive prefix sum
// Strategy:
//   1. Each thread loads 4 elem, computes local prefix (3 adds)
//   2. Warp-scan via SHFL-up on the warp totals (5 levels)
//   3. Cross-warp: warp totals reduced via SMEM + single-warp scan
//   4. Add cross-warp prefix back to each thread's results
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(16) unsigned int smem[1024];
    __shared__ unsigned int warp_totals[8];

    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid & 31;

    smem[tid * 4 + 0] = (tid * 4 + 0 + (unsigned)u2) & 0xFF;
    smem[tid * 4 + 1] = (tid * 4 + 1 + (unsigned)u2) & 0xFF;
    smem[tid * 4 + 2] = (tid * 4 + 2 + (unsigned)u2) & 0xFF;
    smem[tid * 4 + 3] = (tid * 4 + 3 + (unsigned)u2) & 0xFF;
    __syncthreads();

    unsigned long long t0, t1;
    if (tid == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < ITERS; it++) {
        // Step 1: load 4 + local prefix
        unsigned int v0 = smem[tid * 4 + 0];
        unsigned int v1 = smem[tid * 4 + 1] + v0;
        unsigned int v2 = smem[tid * 4 + 2] + v1;
        unsigned int v3 = smem[tid * 4 + 3] + v2;

        // Step 2: warp-scan on per-thread total (= v3) via SHFL-up
        unsigned int warp_total = v3;
        #pragma unroll
        for (int s = 1; s < 32; s *= 2) {
            unsigned int up = __shfl_up_sync(0xFFFFFFFF, warp_total, s);
            if (lane >= s) warp_total += up;
        }
        // warp_total is now inclusive prefix within warp
        // exclusive prefix = warp_total - v3
        unsigned int warp_excl_prefix = warp_total - v3;

        // Step 3: each warp's last lane writes its inclusive total to SMEM
        if (lane == 31) warp_totals[wid] = warp_total;
        __syncthreads();

        // Step 4: warp 0 scans warp_totals (exclusive across warps)
        if (wid == 0) {
            unsigned int wt = (lane < 8) ? warp_totals[lane] : 0;
            unsigned int wt_inc = wt;
            #pragma unroll
            for (int s = 1; s < 8; s *= 2) {
                unsigned int up = __shfl_up_sync(0xFFFFFFFF, wt_inc, s);
                if (lane >= s) wt_inc += up;
            }
            // exclusive: shift by 1
            if (lane < 8) warp_totals[lane] = wt_inc - wt;
        }
        __syncthreads();

        // Step 5: add cross-warp prefix to each result
        unsigned int prefix = warp_totals[wid] + warp_excl_prefix;
        smem[tid * 4 + 0] = v0 + prefix;
        smem[tid * 4 + 1] = v1 + prefix;
        smem[tid * 4 + 2] = v2 + prefix;
        smem[tid * 4 + 3] = v3 + prefix;
        __syncthreads();
    }

    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        printf("Prefix scan 1024 (SHFL warp): clk=%llu cy/scan=%.0f time_ns=%.0f\n",
               t1-t0, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS / 1.5);
    }
    if (smem[1023] == 0xDEADBEEF) C[blockIdx.x] = (float)smem[1023];
}
