// Ninja F: try multiple tricks to win
//   F0: multimem.red (multicast) — sm_90+ ATOMG over multimem
//   F1: red.async (async REDuce) - fire-and-forget without even waiting for in-order
//   F2: atom + ld interleaved using mbarrier wrap?  (skip - mbarrier is CTA-scoped)
//   F3: sense-reversing with PERIODIC reset to keep counter small (= cleaner cache behavior)
//   F4: 2 counters with "complete after both" (split contention but spin only on epoch)
//
// We do F3 here: sense-reversing with smaller counter cycling (so atomic value never grows past 256)
// vs the original uses ((phase+1) * gridSz) which hits 200 * 148 = 29600 - moderate
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef VARIANT
#define VARIANT 0
#endif

// VARIANT 0: red.async fire-and-forget atom with no wait
__device__ __forceinline__ void ninjaF0_sync(unsigned long long* arrival, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        // Fire-and-forget reduction (no return, async if possible)
        asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" :: "l"(arrival) : "memory");
        while (true) {
            unsigned long long cur;
            asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            if (cur >= thresh) break;
        }
    }
    __syncthreads();
}

// VARIANT 1: red.acq_rel — explicit acq_rel for the reduction
__device__ __forceinline__ void ninjaF1_sync(unsigned long long* arrival, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        asm volatile("red.release.gpu.global.add.u64 [%0], 1;" :: "l"(arrival) : "memory");
        while (true) {
            unsigned long long cur;
            asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            if (cur >= thresh) break;
        }
    }
    __syncthreads();
}

// VARIANT 2: red + spin with PAUSE (nanosleep?) backoff
__device__ __forceinline__ void ninjaF2_sync(unsigned long long* arrival, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" :: "l"(arrival) : "memory");
        while (true) {
            unsigned long long cur;
            asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            if (cur >= thresh) break;
            // Light backoff
            asm volatile("nanosleep.u32 16;" ::: "memory");
        }
    }
    __syncthreads();
}

// VARIANT 3: 32-bit counter (lighter than 64-bit)
__device__ __forceinline__ void ninjaF3_sync(unsigned int* arrival32, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int thresh = (unsigned int)((phase + 1) * gridSz);
        asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" :: "l"(arrival32) : "memory");
        while (true) {
            unsigned int cur;
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(cur) : "l"(arrival32) : "memory");
            if (cur >= thresh) break;
        }
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(128, 1)
void kernel_ninja_F(unsigned long long* per_block_min,
                    unsigned long long* per_block_sum,
                    unsigned long long* arrival,
                    unsigned int* arrival32,
                    int* phase_arr,
                    int ITERS) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gridSz = gridDim.x;
    unsigned long long b_min = 0xFFFFFFFFFFFFFFFFULL, b_sum = 0;

    for (int it = 0; it < ITERS; it++) {
        if (tid == 0) phase_arr[bid] = bid + it;
        __syncthreads();
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
#if VARIANT == 0
        ninjaF0_sync(arrival, it, gridSz);
#elif VARIANT == 1
        ninjaF1_sync(arrival, it, gridSz);
#elif VARIANT == 2
        ninjaF2_sync(arrival, it, gridSz);
#elif VARIANT == 3
        ninjaF3_sync(arrival32, it, gridSz);
#endif
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        unsigned long long d = t1 - t0;
        b_sum += d;
        if (d < b_min) b_min = d;
    }
    if (tid == 0) {
        per_block_min[bid] = b_min;
        per_block_sum[bid] = b_sum;
    }
}

int main(int argc, char** argv) {
    int grid = (argc > 1) ? atoi(argv[1]) : 148;
    int iters = (argc > 2) ? atoi(argv[2]) : 200;
    int threads = (argc > 3) ? atoi(argv[3]) : 128;

    CK(cudaSetDevice(0));

    unsigned long long *d_min, *d_sum, *d_arrival;
    unsigned int *d_arr32;
    int *d_phase;
    CK(cudaMalloc(&d_min, grid * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_sum, grid * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_arrival, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_arr32, 4 * sizeof(unsigned int)));
    CK(cudaMalloc(&d_phase, grid * sizeof(int)));
    CK(cudaMemset(d_min, 0xFF, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_sum, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arr32, 0, 4 * sizeof(unsigned int)));
    CK(cudaMemset(d_phase, 0, grid * sizeof(int)));

    kernel_ninja_F<<<grid, threads>>>(d_min, d_sum, d_arrival, d_arr32, d_phase, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_min, 0xFF, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_sum, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arr32, 0, 4 * sizeof(unsigned int)));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    kernel_ninja_F<<<grid, threads>>>(d_min, d_sum, d_arrival, d_arr32, d_phase, iters);
    cudaEventRecord(e1, 0);
    CK(cudaDeviceSynchronize());
    float ms; cudaEventElapsedTime(&ms, e0, e1);

    unsigned long long *h_min = (unsigned long long*)malloc(grid * sizeof(unsigned long long));
    unsigned long long *h_sum = (unsigned long long*)malloc(grid * sizeof(unsigned long long));
    CK(cudaMemcpy(h_min, d_min, grid * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_sum, d_sum, grid * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    double mean = 0;
    unsigned long long mn = 0xFFFFFFFFFFFFFFFFULL;
    for (int b = 0; b < grid; b++) { mean += (double)h_sum[b] / iters; if (h_min[b] < mn) mn = h_min[b]; }
    mean /= grid;
    const char* name[] = {"red.relaxed+ld.relaxed","red.release+ld.acq","red.relaxed+nanosleep","red.relax 32-bit counter"};
    printf("# ninja_F V=%d (%s) grid=%d iters=%d\n", VARIANT, name[VARIANT], grid, iters);
    printf("# mean_cy=%.1f min_cy=%llu wall_us=%.3f\n", mean, mn, (double)ms*1000.0/iters);
    free(h_min); free(h_sum);
    cudaFree(d_min); cudaFree(d_sum); cudaFree(d_arrival); cudaFree(d_arr32); cudaFree(d_phase);
    return 0;
}
