// Ninja D: cluster + global tree (k-ary)
// Hypothesis: B300 has cluster barriers (mbarrier) that work across CTAs in the same cluster.
// We can use a 2-level scheme: cluster-local sync (mbarrier or DSMEM) + a tiny per-cluster
// atomic to a global counter. With cluster size = up to 8 CTAs, we reduce 148 -> ~19 atomics.
//
// Simpler version (without clusters): k-ary tree of 4 — each leader does atomicAdd to one of
// 4 sub-counters (chosen by blockIdx%4), then a designated root spins on all 4 sub-counters.
// This reduces hot-line contention 4×.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define K 4   // number of sub-counters

__device__ __forceinline__ void ninja_D_grid_sync(
    unsigned long long* arrival_k, unsigned long long* epoch, int phase, int gridSz)
{
    __syncthreads();
    if (threadIdx.x == 0) {
        // Per-bucket sense-reversing thresholds. Each bucket k receives sum_{b s.t. b%K==k} 1
        // = ceil((gridSz - k) / K). We want each bucket's count to reach (phase+1)*bucket_size.
        unsigned long long e_target = (unsigned long long)(phase + 1);
        int bucket = blockIdx.x % K;

        if (blockIdx.x != 0) {
            // Worker: relaxed atomic add to its bucket sub-counter, spin on epoch
            unsigned long long dummy;
            asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;"
                         : "=l"(dummy) : "l"(&arrival_k[bucket]) : "memory");
            while (true) {
                unsigned long long cur;
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(epoch) : "memory");
                if (cur >= e_target) break;
            }
        } else {
            // Coordinator (block 0, bucket 0): also adds 1 to bucket 0
            unsigned long long dummy;
            asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;"
                         : "=l"(dummy) : "l"(&arrival_k[0]) : "memory");
            // Spin on all K buckets — each must reach its expected size at this phase
            unsigned long long thresh[K];
            for (int b = 0; b < K; b++) {
                int bsize = (gridSz - b + K - 1) / K;  // ceil((gridSz - b) / K)
                thresh[b] = (unsigned long long)(phase + 1) * (unsigned long long)bsize;
            }
            unsigned int done_mask = 0;
            unsigned int all_mask = (1u << K) - 1u;
            while (done_mask != all_mask) {
                for (int b = 0; b < K; b++) {
                    if (done_mask & (1u << b)) continue;
                    unsigned long long cur;
                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(&arrival_k[b]) : "memory");
                    if (cur >= thresh[b]) done_mask |= (1u << b);
                }
            }
            asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(epoch), "l"(e_target) : "memory");
        }
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(128, 1)
void kernel_ninja_D(unsigned long long* out_cycles,
                    unsigned long long* out_pingpong,
                    unsigned long long* arrival_k,
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
        ninja_D_grid_sync(arrival_k, epoch, it, gridSz);
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
    // Allocate K * 64 bytes apart so each bucket lives on its own L2 line
    CK(cudaMalloc(&d_arrival, 16 * 64));   // way more than enough; align in kernel via index
    CK(cudaMalloc(&d_epoch, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_phase, grid * sizeof(int)));
    CK(cudaMemset(d_out_cy, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_out_pp, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 16 * 64));
    CK(cudaMemset(d_epoch, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_phase, 0, grid * sizeof(int)));

    // For per-cache-line spacing: pass arrival_k as offset 0,8,16,24 - each 8B - all on same line.
    // To put on different lines, we'd index arrival_k[b * 8] (=64B stride). But for the test,
    // start with same-line first. Then run a second variant.

    kernel_ninja_D<<<grid, threads>>>(d_out_cy, d_out_pp, d_arrival, d_epoch, d_phase, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_out_cy, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_out_pp, 0, 8 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 16 * 64));
    CK(cudaMemset(d_epoch, 0, 4 * sizeof(unsigned long long)));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    kernel_ninja_D<<<grid, threads>>>(d_out_cy, d_out_pp, d_arrival, d_epoch, d_phase, iters);
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

    printf("# ninja_D: K=%d sub-counters + epoch broadcast (relaxed atomics)\n", K);
    printf("grid=%d threads=%d iters=%d\n", grid, threads, iters);
    printf("avg_cy_per_sync=%.1f min_cy_per_sync=%.0f wall_us_per_sync=%.3f wall_ms_total=%.3f\n",
           avg_cy, min_cy, wall_us_per_sync, ms);
    printf("pingpong_sumcheck=%llu (anti-DCE)\n", h_pp[0]);

    cudaFree(d_out_cy); cudaFree(d_out_pp); cudaFree(d_arrival); cudaFree(d_epoch); cudaFree(d_phase);
    return 0;
}
