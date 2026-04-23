// Ninja C: split phase counter — separate "I'm done" from "everyone wait"
// Use TWO counters:
//   arrival[] : sense-reversing - block leader atomicAdd's it
//   epoch     : block 0 (after observing arrival reach threshold) writes the new epoch
// All other blocks spin only on `epoch`. This isolates the contended atomic from the broadcast.
//
// Different from ninja A: epoch is a single 8-byte word that increments by 1 per phase.
// Other blocks just compare against expected epoch (=phase+1). Single store + single load (no add).
// Block 0 carries the burden of the spin on the atomic counter.
//
// This is essentially the "central coordinator" pattern, but with the optimization that
// non-coordinator blocks observe a write-only line (light) and the coordinator block's
// arrival-counter spin is on a different cache line that never has many spin readers.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__device__ __forceinline__ void ninja_C_grid_sync(
    unsigned long long* arrival, unsigned long long* epoch, int phase, int gridSz)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long a_thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        unsigned long long e_target = (unsigned long long)(phase + 1);
        if (blockIdx.x != 0) {
            // Workers: relaxed atomic add to arrival counter, then spin on epoch
            unsigned long long dummy;
            asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;"
                         : "=l"(dummy) : "l"(arrival) : "memory");
            while (true) {
                unsigned long long cur;
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(epoch) : "memory");
                if (cur >= e_target) break;
            }
        } else {
            // Coordinator (block 0): also adds 1 to count itself, then spins on arrival
            unsigned long long dummy;
            asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;"
                         : "=l"(dummy) : "l"(arrival) : "memory");
            while (true) {
                unsigned long long cur;
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
                if (cur >= a_thresh) break;
            }
            // All blocks have arrived; release everyone via epoch bump
            asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(epoch), "l"(e_target) : "memory");
        }
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(128, 1)
void kernel_ninja_C(unsigned long long* out_cycles,
                    unsigned long long* out_pingpong,
                    unsigned long long* arrival,
                    unsigned long long* epoch,
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
        ninja_C_grid_sync(arrival, epoch, it, gridSz);
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

    unsigned long long *d_out_cy, *d_out_pp, *d_arrival, *d_epoch;
    int *d_phase;
    CK(cudaMalloc(&d_out_cy, 8 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_out_pp, 8 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_arrival, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_epoch, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_phase, grid * sizeof(int)));
    CK(cudaMemset(d_out_cy, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_out_pp, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_epoch, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_phase, 0, grid * sizeof(int)));

    kernel_ninja_C<<<grid, threads>>>(d_out_cy, d_out_pp, d_arrival, d_epoch, d_phase, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_out_cy, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_out_pp, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_epoch, 0, 4 * sizeof(unsigned long long)));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    kernel_ninja_C<<<grid, threads>>>(d_out_cy, d_out_pp, d_arrival, d_epoch, d_phase, iters);
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

    printf("# ninja_C: split arrival(workers) + epoch(broadcast) - relaxed atomic\n");
    printf("grid=%d threads=%d iters=%d\n", grid, threads, iters);
    printf("avg_cy_per_sync=%.1f min_cy_per_sync=%.0f wall_us_per_sync=%.3f wall_ms_total=%.3f\n",
           avg_cy, min_cy, wall_us_per_sync, ms);
    printf("pingpong_sumcheck=%llu (anti-DCE)\n", h_pp[0]);

    cudaFree(d_out_cy); cudaFree(d_out_pp); cudaFree(d_arrival); cudaFree(d_epoch); cudaFree(d_phase);
    return 0;
}
