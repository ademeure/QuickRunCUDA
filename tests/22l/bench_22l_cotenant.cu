// Investigation 6: grid sync with co-tenant work BEFORE the sync.
// Test the atomic-counter (acq_rel) sync. Pre-work types:
//   COTEN=0 nothing
//   COTEN=1 1024 FFMA chain (compute-bound)
//   COTEN=2 8 cold DRAM loads (memory-bound)
//   COTEN=3 store to a global location (writes that need to be flushed by the sync)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef COTEN
#define COTEN 0
#endif

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

extern "C" __global__ __launch_bounds__(128, 1)
void kernel_cotenant(unsigned long long* per_block_min,
                     unsigned long long* per_block_sum,
                     unsigned long long* arrival,
                     float* dram_data,
                     float* out_data,
                     int* phase_arr,
                     int ITERS) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gridSz = gridDim.x;
    unsigned long long b_min = 0xFFFFFFFFFFFFFFFFULL;
    unsigned long long b_sum = 0;
    float acc = (float)tid * 0.5f;

    for (int it = 0; it < ITERS; it++) {
        if (tid == 0) phase_arr[bid] = bid + it;

#if COTEN == 1
        // 1024 FFMA chain
        #pragma unroll 1
        for (int i = 0; i < 1024; i++) {
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(acc) : "f"(1.0001f), "f"(0.0001f));
        }
        // Force partial use to defeat DCE
        if (acc < -1e30f) out_data[bid * 128 + tid] = acc;
#elif COTEN == 2
        // 8 DRAM loads from a "fresh" location each iteration
        float v = 0;
        size_t base = ((size_t)bid * 16384 + (size_t)it * 8 + tid) % 67108864ULL;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            v += dram_data[base + i * 16];
        }
        if (v > 1e30f) out_data[bid * 128 + tid] = v;
#elif COTEN == 3
        // Each thread stores to its slot
        out_data[bid * 128 + tid] = (float)(it + tid);
#endif

        __syncthreads();
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        atomic_sync(arrival, it, gridSz);
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

        unsigned long long d = t1 - t0;
        b_sum += d;
        if (d < b_min) b_min = d;
    }

    if (tid == 0) {
        per_block_min[bid] = b_min;
        per_block_sum[bid] = b_sum;
    }
    // anti-DCE
    if (acc > 1e30f && tid == 0) out_data[bid] = acc;
}

int main(int argc, char** argv) {
    int grid = (argc > 1) ? atoi(argv[1]) : 148;
    int iters = (argc > 2) ? atoi(argv[2]) : 200;
    int threads = (argc > 3) ? atoi(argv[3]) : 128;

    CK(cudaSetDevice(0));

    unsigned long long *d_min, *d_sum, *d_arrival;
    float *d_dram, *d_out;
    int *d_phase;
    CK(cudaMalloc(&d_min, grid * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_sum, grid * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_arrival, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_dram, 67108864ULL * sizeof(float)));  // 256 MiB
    CK(cudaMalloc(&d_out, grid * 128 * sizeof(float)));
    CK(cudaMalloc(&d_phase, grid * sizeof(int)));
    CK(cudaMemset(d_min, 0xFF, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_sum, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_phase, 0, grid * sizeof(int)));
    CK(cudaMemset(d_dram, 0, 67108864ULL * sizeof(float)));
    CK(cudaMemset(d_out, 0, grid * 128 * sizeof(float)));

    kernel_cotenant<<<grid, threads>>>(d_min, d_sum, d_arrival, d_dram, d_out, d_phase, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_min, 0xFF, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_sum, 0, grid * sizeof(unsigned long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    kernel_cotenant<<<grid, threads>>>(d_min, d_sum, d_arrival, d_dram, d_out, d_phase, iters);
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

    const char* name[] = {"none","1024 FFMA","8 DRAM loads","store all threads"};
    printf("# COTEN=%d (%s) grid=%d iters=%d  mean_cy=%.1f min_cy=%llu  wall_us_per_iter=%.3f\n",
           COTEN, name[COTEN], grid, iters, mean, mn, (double)ms*1000.0/iters);
    free(h_min); free(h_sum);
    cudaFree(d_min); cudaFree(d_sum); cudaFree(d_arrival); cudaFree(d_dram); cudaFree(d_out); cudaFree(d_phase);
    return 0;
}
