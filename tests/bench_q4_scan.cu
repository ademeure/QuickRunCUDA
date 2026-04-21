// Q4: Vectorized prefix scan at SoL
// 1024 ints in SMEM → 1024 prefix sums in same SMEM
// MODE 0: naive Hillis-Steele (log N steps)
// MODE 1: SHFL-up warp scan + per-warp prefix
// MODE 2: redux.sync.add (impossible for scan; use as estimate)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(16) unsigned int smem[1024];
    __shared__ unsigned int warp_sums[8];

    // Init SMEM with thread-id values
    smem[threadIdx.x] = (threadIdx.x + (unsigned)u2) & 0xFF;
    smem[threadIdx.x + 256] = (threadIdx.x + 256 + (unsigned)u2) & 0xFF;
    smem[threadIdx.x + 512] = (threadIdx.x + 512 + (unsigned)u2) & 0xFF;
    smem[threadIdx.x + 768] = (threadIdx.x + 768 + (unsigned)u2) & 0xFF;
    __syncthreads();

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < ITERS; it++) {
#if MODE == 0
        // Hillis-Steele inclusive scan, 4 elements per thread
        // Naive: log(1024) = 10 steps
        for (int stride = 1; stride < 1024; stride *= 2) {
            unsigned int v0 = (threadIdx.x >= stride) ? smem[threadIdx.x - stride] : 0;
            unsigned int v1 = (threadIdx.x + 256 >= stride) ? smem[threadIdx.x + 256 - stride] : 0;
            __syncthreads();
            smem[threadIdx.x] += v0;
            smem[threadIdx.x + 256] += v1;
            __syncthreads();
        }
#elif MODE == 1
        // Warp-level SHFL-up scan + per-warp prefix
        // Each thread handles 4 ints (uint4)
        int tid = threadIdx.x;
        int wid = tid / 32;
        int lane = tid & 31;
        unsigned int v[4];
        v[0] = smem[tid * 4];
        v[1] = smem[tid * 4 + 1];
        v[2] = smem[tid * 4 + 2];
        v[3] = smem[tid * 4 + 3];
        // Sequential prefix within thread
        v[1] += v[0]; v[2] += v[1]; v[3] += v[2];
        // Warp scan via SHFL (Kogge-Stone)
        unsigned int x = v[3];
        #pragma unroll
        for (int s = 1; s < 32; s *= 2) {
            unsigned int y = __shfl_up_sync(0xFFFFFFFF, x, s);
            if (lane >= s) x += y;
        }
        // x now = inclusive prefix of 4-tuple sums within warp
        unsigned int my_warp_prefix = (lane > 0) ? __shfl_up_sync(0xFFFFFFFF, x, 1) : 0;
        // Now propagate warp totals
        if (lane == 31) warp_sums[wid] = x;
        __syncthreads();
        // Single warp processes warp_sums prefix
        if (wid == 0) {
            unsigned int s = (lane < 8) ? warp_sums[lane] : 0;
            #pragma unroll
            for (int st = 1; st < 8; st *= 2) {
                unsigned int y = __shfl_up_sync(0xFFFFFFFF, s, st);
                if (lane >= st) s += y;
            }
            if (lane < 8) warp_sums[lane] = s;
        }
        __syncthreads();
        unsigned int wprev = (wid > 0) ? warp_sums[wid - 1] : 0;
        unsigned int total_prefix = my_warp_prefix + wprev;
        // Write back inclusive prefix
        smem[tid * 4 + 0] = v[0] + total_prefix;
        smem[tid * 4 + 1] = v[1] + total_prefix;
        smem[tid * 4 + 2] = v[2] + total_prefix;
        smem[tid * 4 + 3] = v[3] + total_prefix;
        __syncthreads();
#endif
    }

    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        printf("MODE=%d clk=%llu cy/scan=%.0f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
    if (smem[1023] == 0xDEADBEEF) C[blockIdx.x] = (float)smem[1023];
}
