// Investigation 1: NVIDIA cooperative_groups::grid_group::sync()
// Launched via cudaLaunchCooperativeKernel.
// Measures cycles using clock64 around the sync inside the kernel.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// ITERS: number of grid syncs measured.
// We use clock64 around each sync and accumulate min/sum of per-sync deltas.
// Anti-DCE: write the accumulated count + ping-pong reduction result to out.
extern "C" __global__ __launch_bounds__(128, 1)
void kernel_cg_sync(unsigned long long* out_cycles, unsigned long long* out_pingpong, int ITERS, int* phase) {
    auto grid = cg::this_grid();

    // ping-pong reduction sanity workload
    // Each iteration: each block writes blockIdx.x to phase[blockIdx.x]
    // grid.sync()
    // Each block reads sum of phase[0..gridDim-1] (only block 0 thread 0 actually computes the sum)
    unsigned long long sum_cycles = 0;
    unsigned long long min_cycles = 0xFFFFFFFFFFFFFFFFULL;
    unsigned long long sumcheck = 0;

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gridSz = gridDim.x;

    for (int it = 0; it < ITERS; it++) {
        // producer phase: block leader writes its iteration tag
        if (tid == 0) phase[bid] = bid + it;
        __syncthreads();

        unsigned long long t0, t1;
        // Use volatile asm to prevent reorder
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        grid.sync();
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

        unsigned long long d = t1 - t0;
        sum_cycles += d;
        if (d < min_cycles) min_cycles = d;

        // consumer phase: block 0 thread 0 sums all phases
        if (bid == 0 && tid == 0) {
            unsigned long long s = 0;
            for (int b = 0; b < gridSz; b++) s += (unsigned long long)phase[b];
            sumcheck += s;
        }
    }

    if (bid == 0 && tid == 0) {
        out_cycles[0] = sum_cycles;
        out_cycles[1] = min_cycles;
        out_cycles[2] = (unsigned long long)ITERS;
        out_pingpong[0] = sumcheck;
    }
}

int main(int argc, char** argv) {
    int grid = (argc > 1) ? atoi(argv[1]) : 148;
    int iters = (argc > 2) ? atoi(argv[2]) : 200;
    int threads = (argc > 3) ? atoi(argv[3]) : 128;

    CK(cudaSetDevice(0));

    int coop_supported;
    CK(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, 0));
    if (!coop_supported) { printf("# coop launch not supported\n"); return 1; }

    int max_blocks_per_sm;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel_cg_sync, threads, 0));
    int max_grid_for_coop = 148 * max_blocks_per_sm;
    if (grid > max_grid_for_coop) {
        printf("# requested grid %d > max_coop %d (occupancy=%d/SM)\n", grid, max_grid_for_coop, max_blocks_per_sm);
        return 1;
    }

    unsigned long long *d_out_cy, *d_out_pp;
    int *d_phase;
    CK(cudaMalloc(&d_out_cy, 8 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_out_pp, 8 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_phase, grid * sizeof(int)));
    CK(cudaMemset(d_out_cy, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_out_pp, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_phase, 0, grid * sizeof(int)));

    void* args[] = { &d_out_cy, &d_out_pp, &iters, &d_phase };

    // warmup
    CK(cudaLaunchCooperativeKernel((void*)kernel_cg_sync, dim3(grid), dim3(threads), args, 0, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_out_cy, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_out_pp, 0, 8 * sizeof(unsigned long long)));

    // Wall clock event timing
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    CK(cudaLaunchCooperativeKernel((void*)kernel_cg_sync, dim3(grid), dim3(threads), args, 0, 0));
    cudaEventRecord(e1, 0);
    CK(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);

    unsigned long long h_out[8];
    CK(cudaMemcpy(h_out, d_out_cy, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    unsigned long long h_pp[8];
    CK(cudaMemcpy(h_pp, d_out_pp, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    double avg_cy = (double)h_out[0] / (double)iters;
    double min_cy = (double)h_out[1];
    double wall_us_per_sync = (double)ms * 1000.0 / (double)iters;

    printf("# cg::grid_group::sync()\n");
    printf("grid=%d threads=%d iters=%d max_blocks_per_sm=%d\n",
           grid, threads, iters, max_blocks_per_sm);
    printf("avg_cy_per_sync=%.1f min_cy_per_sync=%.0f wall_us_per_sync=%.3f wall_ms_total=%.3f\n",
           avg_cy, min_cy, wall_us_per_sync, ms);
    printf("pingpong_sumcheck=%llu (anti-DCE)\n", h_pp[0]);

    cudaFree(d_out_cy); cudaFree(d_out_pp); cudaFree(d_phase);
    return 0;
}
