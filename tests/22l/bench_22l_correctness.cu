// Correctness verification: ping-pong reduction across grid syncs
// Workload that REQUIRES correct grid sync semantics:
//   Phase 0: each block writes blockIdx.x to data[blockIdx.x]
//   grid_sync()
//   Phase 1: each block reads sum of data[0..gridDim-1] AND writes (sum + blockIdx.x) to result[blockIdx.x]
//   grid_sync()
//   Phase 2: read sum of result[]
// Expected sum = gridDim * (gridDim*(gridDim-1)/2) + (gridDim*(gridDim-1)/2)
//              = (gridDim+1) * (gridDim*(gridDim-1)/2)
// Modes: same as bench_22l_perblock.cu
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
        asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;" : "=l"(dummy) : "l"(arrival) : "memory");
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
            asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;" : "=l"(dummy) : "l"(arrival) : "memory");
            while (true) {
                unsigned long long cur;
                asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(epoch) : "memory");
                if (cur >= e_target) break;
            }
        } else {
            unsigned long long a_thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
            unsigned long long dummy;
            asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;" : "=l"(dummy) : "l"(arrival) : "memory");
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
void kernel_pingpong(unsigned long long* arrival,
                     unsigned long long* epoch,
                     long long* data,
                     long long* result,
                     long long* final_sum) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gridSz = gridDim.x;
#if MODE == 0
    auto grid = cg::this_grid();
#endif

    // Phase 0: each block writes bid to data[bid]
    if (tid == 0) data[bid] = (long long)bid;

#if MODE == 0
    grid.sync();
#elif MODE == 1
    atomic_sync(arrival, 0, gridSz);
#elif MODE == 2
    ninjaB_sync(arrival, 0, gridSz);
#elif MODE == 3
    ninjaC_sync(arrival, epoch, 0, gridSz);
#endif

    // Phase 1: each block reads sum of data[]; writes sum+bid to result[bid]
    if (tid == 0) {
        long long s = 0;
        for (int b = 0; b < gridSz; b++) s += data[b];
        result[bid] = s + bid;
    }

#if MODE == 0
    grid.sync();
#elif MODE == 1
    atomic_sync(arrival, 1, gridSz);
#elif MODE == 2
    ninjaB_sync(arrival, 1, gridSz);
#elif MODE == 3
    ninjaC_sync(arrival, epoch, 1, gridSz);
#endif

    // Phase 2: only block 0 reads sum of result[]
    if (bid == 0 && tid == 0) {
        long long s = 0;
        for (int b = 0; b < gridSz; b++) s += result[b];
        final_sum[0] = s;
    }
}

int main(int argc, char** argv) {
    int grid = (argc > 1) ? atoi(argv[1]) : 148;
    int threads = (argc > 2) ? atoi(argv[2]) : 128;

    CK(cudaSetDevice(0));

    unsigned long long *d_arrival, *d_epoch;
    long long *d_data, *d_result, *d_final;
    CK(cudaMalloc(&d_arrival, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_epoch, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_data, grid * sizeof(long long)));
    CK(cudaMalloc(&d_result, grid * sizeof(long long)));
    CK(cudaMalloc(&d_final, sizeof(long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_epoch, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_data, 0, grid * sizeof(long long)));
    CK(cudaMemset(d_result, 0, grid * sizeof(long long)));
    CK(cudaMemset(d_final, 0, sizeof(long long)));

    void* args[] = { &d_arrival, &d_epoch, &d_data, &d_result, &d_final };

#if MODE == 0
    CK(cudaLaunchCooperativeKernel((void*)kernel_pingpong, dim3(grid), dim3(threads), args, 0, 0));
#else
    kernel_pingpong<<<grid, threads>>>(d_arrival, d_epoch, d_data, d_result, d_final);
#endif
    CK(cudaDeviceSynchronize());

    long long h_final = 0;
    CK(cudaMemcpy(&h_final, d_final, sizeof(long long), cudaMemcpyDeviceToHost));
    long long expected = (long long)(grid + 1) * (long long)grid * (long long)(grid - 1) / 2;
    const char* name[] = {"cg::grid.sync", "atomic_acqrel", "ninja_B_relaxed", "ninja_C_split"};
    printf("MODE=%d (%s) grid=%d  got=%lld expected=%lld %s\n",
           MODE, name[MODE], grid, h_final, expected,
           (h_final == expected) ? "PASS" : "FAIL");

    cudaFree(d_arrival); cudaFree(d_epoch); cudaFree(d_data); cudaFree(d_result); cudaFree(d_final);
    return (h_final == expected) ? 0 : 1;
}
