// Per-block cycle measurement: each block writes its own min/max/sum cycles
// Tests: cg, atomic, ninja_B, ninja_C variants — selectable via MODE.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef MODE
#define MODE 0
#endif

// Mode 0: cg::sync.  Mode 1: atomic counter (sense-reversing, acq_rel).
// Mode 2: ninja_B (relaxed + fence). Mode 3: ninja_C (split arrival/epoch).
__device__ __forceinline__ void atomic_sync(unsigned long long* arrival, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        atomicAdd(arrival, 1ULL);
        while (true) {
            unsigned long long cur;
            asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            if (cur >= thresh) break;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void ninjaB_sync(unsigned long long* arrival, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        asm volatile("fence.release.gpu;" ::: "memory");
        unsigned long long dummy;
        asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;"
                     : "=l"(dummy) : "l"(arrival) : "memory");
        while (true) {
            unsigned long long cur;
            asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            if (cur >= thresh) break;
        }
        asm volatile("fence.acquire.gpu;" ::: "memory");
    }
    __syncthreads();
}

__device__ __forceinline__ void ninjaC_sync(unsigned long long* arrival, unsigned long long* epoch, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long e_target = (unsigned long long)(phase + 1);
        if (blockIdx.x != 0) {
            unsigned long long dummy;
            asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;"
                         : "=l"(dummy) : "l"(arrival) : "memory");
            while (true) {
                unsigned long long cur;
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(epoch) : "memory");
                if (cur >= e_target) break;
            }
        } else {
            unsigned long long a_thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
            unsigned long long dummy;
            asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;"
                         : "=l"(dummy) : "l"(arrival) : "memory");
            while (true) {
                unsigned long long cur;
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
                if (cur >= a_thresh) break;
            }
            asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(epoch), "l"(e_target) : "memory");
        }
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(128, 1)
void kernel_perblock(unsigned long long* per_block_min,
                     unsigned long long* per_block_max,
                     unsigned long long* per_block_sum,
                     unsigned long long* arrival,
                     unsigned long long* epoch,
                     int* phase_arr,
                     int ITERS) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gridSz = gridDim.x;
#if MODE == 0
    auto grid = cg::this_grid();
#endif

    unsigned long long b_min = 0xFFFFFFFFFFFFFFFFULL;
    unsigned long long b_max = 0;
    unsigned long long b_sum = 0;

    for (int it = 0; it < ITERS; it++) {
        if (tid == 0) phase_arr[bid] = bid + it;
        __syncthreads();

        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
#if MODE == 0
        grid.sync();
#elif MODE == 1
        atomic_sync(arrival, it, gridSz);
#elif MODE == 2
        ninjaB_sync(arrival, it, gridSz);
#elif MODE == 3
        ninjaC_sync(arrival, epoch, it, gridSz);
#endif
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

        unsigned long long d = t1 - t0;
        b_sum += d;
        if (d < b_min) b_min = d;
        if (d > b_max) b_max = d;
    }

    if (tid == 0) {
        per_block_min[bid] = b_min;
        per_block_max[bid] = b_max;
        per_block_sum[bid] = b_sum;
    }
}

int main(int argc, char** argv) {
    int grid = (argc > 1) ? atoi(argv[1]) : 148;
    int iters = (argc > 2) ? atoi(argv[2]) : 200;
    int threads = (argc > 3) ? atoi(argv[3]) : 128;

    CK(cudaSetDevice(0));

    unsigned long long *d_min, *d_max, *d_sum, *d_arrival, *d_epoch;
    int *d_phase;
    CK(cudaMalloc(&d_min, grid * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_max, grid * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_sum, grid * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_arrival, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_epoch, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_phase, grid * sizeof(int)));
    CK(cudaMemset(d_min, 0xFF, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_max, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_sum, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_epoch, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_phase, 0, grid * sizeof(int)));

    void* args[] = { &d_min, &d_max, &d_sum, &d_arrival, &d_epoch, &d_phase, &iters };

    // warmup
#if MODE == 0
    CK(cudaLaunchCooperativeKernel((void*)kernel_perblock, dim3(grid), dim3(threads), args, 0, 0));
#else
    kernel_perblock<<<grid, threads>>>(d_min, d_max, d_sum, d_arrival, d_epoch, d_phase, iters);
#endif
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_min, 0xFF, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_max, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_sum, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_epoch, 0, 4 * sizeof(unsigned long long)));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
#if MODE == 0
    CK(cudaLaunchCooperativeKernel((void*)kernel_perblock, dim3(grid), dim3(threads), args, 0, 0));
#else
    kernel_perblock<<<grid, threads>>>(d_min, d_max, d_sum, d_arrival, d_epoch, d_phase, iters);
#endif
    cudaEventRecord(e1, 0);
    CK(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);

    unsigned long long *h_min = (unsigned long long*)malloc(grid * sizeof(unsigned long long));
    unsigned long long *h_max = (unsigned long long*)malloc(grid * sizeof(unsigned long long));
    unsigned long long *h_sum = (unsigned long long*)malloc(grid * sizeof(unsigned long long));
    CK(cudaMemcpy(h_min, d_min, grid * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_max, d_max, grid * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_sum, d_sum, grid * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // Aggregate stats: per-block avg, then min/avg/max across blocks
    double across_min_avg = 1e30, across_max_avg = 0, across_sum_avg = 0;
    unsigned long long across_min_min = 0xFFFFFFFFFFFFFFFFULL;
    unsigned long long across_max_max = 0;
    int who_min = -1, who_max = -1;

    for (int b = 0; b < grid; b++) {
        double avg = (double)h_sum[b] / iters;
        across_sum_avg += avg;
        if (avg < across_min_avg) { across_min_avg = avg; who_min = b; }
        if (avg > across_max_avg) { across_max_avg = avg; who_max = b; }
        if (h_min[b] < across_min_min) across_min_min = h_min[b];
        if (h_max[b] > across_max_max) across_max_max = h_max[b];
    }
    double mean_avg = across_sum_avg / grid;

    double wall_us_per_sync = (double)ms * 1000.0 / iters;
    const char* name[] = {"cg::grid.sync", "atomic_acqrel", "ninja_B_relaxed", "ninja_C_split"};
    printf("# MODE=%d (%s) grid=%d threads=%d iters=%d\n", MODE, name[MODE], grid, threads, iters);
    printf("# per-block-avg cy: min=%.0f (block %d)  mean=%.0f  max=%.0f (block %d)\n",
           across_min_avg, who_min, mean_avg, across_max_avg, who_max);
    printf("# per-block-min cy across grid: %llu  per-block-max cy: %llu\n", across_min_min, across_max_max);
    printf("# wall_us_per_sync=%.3f wall_ms_total=%.3f\n", wall_us_per_sync, ms);

    free(h_min); free(h_max); free(h_sum);
    cudaFree(d_min); cudaFree(d_max); cudaFree(d_sum);
    cudaFree(d_arrival); cudaFree(d_epoch); cudaFree(d_phase);
    return 0;
}
