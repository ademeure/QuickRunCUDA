// Correctness test for ninja_E and ninja_F0 (the winners)
// Same ping-pong reduction protocol as bench_22l_correctness.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef MODE
#define MODE 0
#endif

// MODE 0: ninja_E (relaxed atom + relaxed spin, no fence)
// MODE 1: ninja_F0 (red.relaxed + relaxed spin)
// MODE 2: ninja_F3 (32-bit counter)
__device__ __forceinline__ void ninjaE_sync(unsigned long long* arrival, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        unsigned long long dummy;
        asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], 1;" : "=l"(dummy) : "l"(arrival) : "memory");
        while (true) {
            unsigned long long cur;
            asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            if (cur >= thresh) break;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void ninjaF0_sync(unsigned long long* arrival, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long thresh = (unsigned long long)(phase + 1) * (unsigned long long)gridSz;
        asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" :: "l"(arrival) : "memory");
        while (true) {
            unsigned long long cur;
            asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(cur) : "l"(arrival) : "memory");
            if (cur >= thresh) break;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void ninjaF3_sync(unsigned int* arrival, int phase, int gridSz) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int thresh = (unsigned int)((phase + 1) * gridSz);
        asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" :: "l"(arrival) : "memory");
        while (true) {
            unsigned int cur;
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(cur) : "l"(arrival) : "memory");
            if (cur >= thresh) break;
        }
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(128, 1)
void kernel_pingpong2(unsigned long long* arrival,
                      unsigned int* arrival32,
                      long long* data,
                      long long* result,
                      long long* final_sum) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int gridSz = gridDim.x;

    if (tid == 0) data[bid] = (long long)bid;

#if MODE == 0
    ninjaE_sync(arrival, 0, gridSz);
#elif MODE == 1
    ninjaF0_sync(arrival, 0, gridSz);
#elif MODE == 2
    ninjaF3_sync(arrival32, 0, gridSz);
#endif

    if (tid == 0) {
        long long s = 0;
        for (int b = 0; b < gridSz; b++) s += data[b];
        result[bid] = s + bid;
    }

#if MODE == 0
    ninjaE_sync(arrival, 1, gridSz);
#elif MODE == 1
    ninjaF0_sync(arrival, 1, gridSz);
#elif MODE == 2
    ninjaF3_sync(arrival32, 1, gridSz);
#endif

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

    unsigned long long *d_arrival;
    unsigned int *d_arr32;
    long long *d_data, *d_result, *d_final;
    CK(cudaMalloc(&d_arrival, 4 * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_arr32, 4 * sizeof(unsigned int)));
    CK(cudaMalloc(&d_data, grid * sizeof(long long)));
    CK(cudaMalloc(&d_result, grid * sizeof(long long)));
    CK(cudaMalloc(&d_final, sizeof(long long)));
    CK(cudaMemset(d_arrival, 0, 4 * sizeof(unsigned long long)));
    CK(cudaMemset(d_arr32, 0, 4 * sizeof(unsigned int)));
    CK(cudaMemset(d_data, 0, grid * sizeof(long long)));
    CK(cudaMemset(d_result, 0, grid * sizeof(long long)));
    CK(cudaMemset(d_final, 0, sizeof(long long)));

    kernel_pingpong2<<<grid, threads>>>(d_arrival, d_arr32, d_data, d_result, d_final);
    CK(cudaDeviceSynchronize());

    long long h_final = 0;
    CK(cudaMemcpy(&h_final, d_final, sizeof(long long), cudaMemcpyDeviceToHost));
    long long expected = (long long)(grid + 1) * (long long)grid * (long long)(grid - 1) / 2;
    const char* name[] = {"ninja_E", "ninja_F0", "ninja_F3 (32-bit)"};
    printf("MODE=%d (%s) grid=%d  got=%lld expected=%lld %s\n",
           MODE, name[MODE], grid, h_final, expected,
           (h_final == expected) ? "PASS" : "FAIL");
    cudaFree(d_arrival); cudaFree(d_arr32); cudaFree(d_data); cudaFree(d_result); cudaFree(d_final);
    return (h_final == expected) ? 0 : 1;
}
