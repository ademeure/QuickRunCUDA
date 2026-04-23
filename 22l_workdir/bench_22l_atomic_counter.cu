// Investigation 2: catalog's "global atomic counter" pattern
// per-block leader does atomicAdd to phase counter, spins until counter == phase * gridDim.
// Sense-reversing scheme: counter monotonically increases; threshold = (phase+1)*gridSz.
// Avoids the reset race issue.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Sense-reversing atomic-counter grid sync.
// arrival_count is monotonic; sync(phase) waits for arrival_count >= (phase+1) * gridSz
__device__ __forceinline__ void atomic_grid_sync(unsigned long long* arrival_count, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        // acq_rel atomic add
        unsigned long long thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        atomicAdd(arrival_count, 1ULL);
        // Spin until all blocks arrived
        while (true) {
            unsigned long long cur;
            asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival_count) : "memory");
            if (cur >= thresh) break;
        }
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(128, 1)
void kernel_atomic_sync(unsigned long long* out_cycles,
                        unsigned long long* out_pingpong,
                        unsigned long long* arrival_count,
                        int* phase_arr,
                        int ITERS) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gridSz = gridDim.x;

    unsigned long long sum_cycles = 0;
    unsigned long long min_cycles = 0xFFFFFFFFFFFFFFFFULL;
    unsigned long long sumcheck = 0;

    for (int it = 0; it < ITERS; it++) {
        if (tid == 0) phase_arr[bid] = bid + it;
        __syncthreads();

        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        atomic_grid_sync(arrival_count, it, gridSz);
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

        unsigned long long d = t1 - t0;
        sum_cycles += d;
        if (d < min_cycles) min_cycles = d;

        if (bid == 0 && tid == 0) {
            unsigned long long s = 0;
            for (int b = 0; b < gridSz; b++) s += (unsigned long long)phase_arr[b];
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

    unsigned long long *d_out_cy, *d_out_pp, *d_arrival;
    int *d_phase;
    CK(cudaMalloc(&d_out_cy, 8 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_out_pp, 8 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_arrival, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_phase, grid * sizeof(int)));
    CK(cudaMemset(d_out_cy, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_out_pp, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_phase, 0, grid * sizeof(int)));

    // warmup
    kernel_atomic_sync<<<grid, threads>>>(d_out_cy, d_out_pp, d_arrival, d_phase, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_out_cy, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_out_pp, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    kernel_atomic_sync<<<grid, threads>>>(d_out_cy, d_out_pp, d_arrival, d_phase, iters);
    cudaEventRecord(e1, 0);
    CK(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);

    unsigned long long h_out[8], h_pp[8];
    CK(cudaMemcpy(h_out, d_out_cy, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_pp, d_out_pp, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    double avg_cy = (double)h_out[0] / (double)iters;
    double min_cy = (double)h_out[1];
    double wall_us_per_sync = (double)ms * 1000.0 / (double)iters;

    printf("# atomic_grid_sync (sense-reversing acq_rel, monotonic counter)\n");
    printf("grid=%d threads=%d iters=%d\n", grid, threads, iters);
    printf("avg_cy_per_sync=%.1f min_cy_per_sync=%.0f wall_us_per_sync=%.3f wall_ms_total=%.3f\n",
           avg_cy, min_cy, wall_us_per_sync, ms);
    printf("pingpong_sumcheck=%llu (anti-DCE)\n", h_pp[0]);

    cudaFree(d_out_cy); cudaFree(d_out_pp); cudaFree(d_arrival); cudaFree(d_phase);
    return 0;
}
