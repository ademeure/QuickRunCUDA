// Decomposition: measure the COST of each component of the atomic-counter sync separately.
// Components:
//   COMP=0 baseline: empty (just clock64 t0/t1 -- noise floor)
//   COMP=1 fence.acquire.gpu only
//   COMP=2 fence.release.gpu only
//   COMP=3 fence.acq_rel.gpu (=acquire+release combo)
//   COMP=4 atom.relaxed.gpu add (no spin, no fence)
//   COMP=5 atom.acq_rel.gpu add (=atomicAdd default)  - hot line uncontended
//   COMP=6 ld.acquire.gpu single read (just one acquire load)
//   COMP=7 ld.relaxed.gpu single read
//   COMP=8 __syncthreads (single)
//   COMP=9 atomic add + ONE acq load + breakout (no waiting at all)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef COMP
#define COMP 0
#endif

extern "C" __global__ __launch_bounds__(128, 1)
void kernel_decomp(unsigned long long* per_block_min,
                   unsigned long long* per_block_sum,
                   unsigned long long* arrival,
                   int* phase_arr,
                   int ITERS) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    unsigned long long b_min = 0xFFFFFFFFFFFFFFFFULL;
    unsigned long long b_sum = 0;

    for (int it = 0; it < ITERS; it++) {
        if (tid == 0) phase_arr[bid] = bid + it;
        __syncthreads();

        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
#if COMP == 0
        // baseline: empty
#elif COMP == 1
        if (tid == 0) asm volatile("fence.acquire.gpu;" ::: "memory");
#elif COMP == 2
        if (tid == 0) asm volatile("fence.release.gpu;" ::: "memory");
#elif COMP == 3
        if (tid == 0) asm volatile("fence.acq_rel.gpu;" ::: "memory");
#elif COMP == 4
        if (tid == 0) {
            unsigned long long dummy;
            asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;" : "=l"(dummy) : "l"(arrival) : "memory");
        }
#elif COMP == 5
        if (tid == 0) {
            unsigned long long dummy;
            asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;" : "=l"(dummy) : "l"(arrival) : "memory");
        }
#elif COMP == 6
        if (tid == 0) {
            unsigned long long cur;
            asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            phase_arr[bid] = (int)cur;
        }
#elif COMP == 7
        if (tid == 0) {
            unsigned long long cur;
            asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            phase_arr[bid] = (int)cur;
        }
#elif COMP == 8
        __syncthreads();
#elif COMP == 9
        // atomicAdd + ONE acq load (no spin)
        if (tid == 0) {
            unsigned long long dummy;
            asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;" : "=l"(dummy) : "l"(arrival) : "memory");
            unsigned long long cur;
            asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            phase_arr[bid] = (int)cur;
        }
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
    int iters = (argc > 2) ? atoi(argv[2]) : 1000;
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

    kernel_decomp<<<grid, threads>>>(d_min, d_sum, d_arrival, d_phase, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_min, 0xFF, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_sum, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    kernel_decomp<<<grid, threads>>>(d_min, d_sum, d_arrival, d_phase, iters);
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

    const char* name[] = {"baseline","fence.acq","fence.rel","fence.acq_rel","atom.relax","atom.acqrel","ld.acq","ld.rel","__syncthreads","atom+ld(no spin)"};
    printf("# COMP=%d (%s) grid=%d iters=%d  mean_cy=%.1f  min_cy=%llu  wall_us=%.3f\n",
           COMP, name[COMP], grid, iters, mean, mn, (double)ms*1000.0/iters);
    free(h_min); free(h_sum);
    cudaFree(d_min); cudaFree(d_sum); cudaFree(d_arrival); cudaFree(d_phase);
    return 0;
}
