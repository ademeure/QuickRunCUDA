// Ninja E: TRULY relaxed atom + spin loop, NO fence at all.
// Hypothesis: for sense-reversing counter sync where data is already coherent (single counter,
// monotonic), we don't need ANY fence — the spin's acquire load is the only ordering needed.
// This is unsafe IF other data must be visible to peers (which it usually does in real apps),
// but as a baseline cost measurement, it shows the floor.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__device__ __forceinline__ void ninjaE_sync(unsigned long long* arrival, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        // Pure relaxed atom add (no fence)
        unsigned long long dummy;
        asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;" : "=l"(dummy) : "l"(arrival) : "memory");
        // Spin with relaxed loads + nonatomic compare
        while (true) {
            unsigned long long cur;
            asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            if (cur >= thresh) break;
        }
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(128, 1)
void kernel_ninja_E(unsigned long long* per_block_min,
                    unsigned long long* per_block_sum,
                    unsigned long long* arrival,
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
        ninjaE_sync(arrival, it, gridSz);
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
    int *d_phase;
    CK(cudaMalloc(&d_min, grid * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_sum, grid * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_arrival, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_phase, grid * sizeof(int)));
    CK(cudaMemset(d_min, 0xFF, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_sum, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_phase, 0, grid * sizeof(int)));

    kernel_ninja_E<<<grid, threads>>>(d_min, d_sum, d_arrival, d_phase, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_min, 0xFF, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_sum, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    kernel_ninja_E<<<grid, threads>>>(d_min, d_sum, d_arrival, d_phase, iters);
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

    printf("# ninja_E: pure relaxed atom + relaxed spin (NO FENCE) grid=%d iters=%d\n", grid, iters);
    printf("# mean_cy=%.1f min_cy=%llu wall_us_per_sync=%.3f\n", mean, mn, (double)ms*1000.0/iters);
    free(h_min); free(h_sum);
    cudaFree(d_min); cudaFree(d_sum); cudaFree(d_arrival); cudaFree(d_phase);
    return 0;
}
