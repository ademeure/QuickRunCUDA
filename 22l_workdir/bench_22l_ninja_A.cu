// Ninja A: single-thread fence
// Hypothesis: only block leader needs to do the atomic; the membar is implicit in acq/rel semantics
// at the leader. All other threads just __syncthreads — and most of the wait time is *idle spin*,
// so we can let MULTIPLE block-leaders share the same counter polling but spread spin reads
// across only block 0 thread 0 to reduce HBM/L2 polling traffic.
//
// Variant: "polling broadcast" — only thread 0 of block 0 spins on the counter; once it sees
// arrival_count >= threshold, it sets a shared "release" flag. All other block leaders spin
// on that release flag (which lives in a hot L2 line, lighter weight than the counter).
//
// This trades 1 polling line for 1 shared release line. Net: 1 producer + N consumers, each
// reading 1 cache line.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Sense-reversing arrival counter + separate release counter.
// Block leaders all atomicAdd to arrival; block 0's leader spins until threshold, then bumps release.
// All other block leaders spin on release.
__device__ __forceinline__ void ninja_A_grid_sync(
    unsigned long long* arrival, unsigned long long* release, int phase, int gridSz)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long a_thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        unsigned long long r_thresh = (unsigned long long)(phase + 1);
        // Producer side: every block does the atomicAdd
        atomicAdd(arrival, 1ULL);

        if (blockIdx.x == 0) {
            // Block 0 leader: spin on arrival, then bump release
            while (true) {
                unsigned long long cur;
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
                if (cur >= a_thresh) break;
            }
            // Release everyone (relaxed store - producers can't have already seen our update if it's not present yet)
            asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(release), "l"(r_thresh) : "memory");
        } else {
            // Other block leaders: spin on release only (much lighter than counter)
            while (true) {
                unsigned long long cur;
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(release) : "memory");
                if (cur >= r_thresh) break;
            }
        }
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(128, 1)
void kernel_ninja_A(unsigned long long* out_cycles,
                    unsigned long long* out_pingpong,
                    unsigned long long* arrival,
                    unsigned long long* release,
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
        ninja_A_grid_sync(arrival, release, it, gridSz);
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

    unsigned long long *d_out_cy, *d_out_pp, *d_arrival, *d_release;
    int *d_phase;
    CK(cudaMalloc(&d_out_cy, 8 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_out_pp, 8 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_arrival, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_release, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_phase, grid * sizeof(int)));
    CK(cudaMemset(d_out_cy, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_out_pp, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_release, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_phase, 0, grid * sizeof(int)));

    kernel_ninja_A<<<grid, threads>>>(d_out_cy, d_out_pp, d_arrival, d_release, d_phase, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_out_cy, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_out_pp, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_release, 0, 4 * sizeof(unsigned long long)));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    kernel_ninja_A<<<grid, threads>>>(d_out_cy, d_out_pp, d_arrival, d_release, d_phase, iters);
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

    printf("# ninja_A: split arrival/release (block 0 leader is the relay)\n");
    printf("grid=%d threads=%d iters=%d\n", grid, threads, iters);
    printf("avg_cy_per_sync=%.1f min_cy_per_sync=%.0f wall_us_per_sync=%.3f wall_ms_total=%.3f\n",
           avg_cy, min_cy, wall_us_per_sync, ms);
    printf("pingpong_sumcheck=%llu (anti-DCE)\n", h_pp[0]);

    cudaFree(d_out_cy); cudaFree(d_out_pp); cudaFree(d_arrival); cudaFree(d_release); cudaFree(d_phase);
    return 0;
}
