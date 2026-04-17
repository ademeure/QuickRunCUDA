// REAL benefit: 4 blocks of 57 KiB (trick) vs 3 blocks of 57 KiB (no trick)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define ITERS 5000
#define SHMEM_USE_BYTES (57 * 1024)  // we want 57 KiB usable

// Without trick: declare 57 KiB → only 3 blocks/SM
extern "C" __global__ void k_no_trick(float *out) {
    extern __shared__ float buf[];  // 57 KiB
    int tid = threadIdx.x;
    // Init shmem
    int floats = SHMEM_USE_BYTES / 4;
    for (int i = tid; i < floats; i += blockDim.x)
        buf[i] = (float)(i * 0.001f);
    __syncthreads();

    // Compute using shmem
    float a = 1.0f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a += buf[(tid + i) % floats];
    }
    if (tid == 0) out[blockIdx.x] = a;
}

// With trick: declare 56 KiB compiler, use 1 KiB stolen
extern "C" __global__ void k_with_trick(float *out) {
    extern __shared__ float buf[];  // 56 KiB compiler-aware
    int tid = threadIdx.x;

    // Init compiler-aware (56 KiB = 14336 floats)
    for (int i = tid; i < 14336; i += blockDim.x)
        buf[i] = (float)(i * 0.001f);

    // Init stolen (1 KiB = 256 floats at offset 0..1023)
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val_int = __float_as_int((float)((14336 + i) * 0.001f));
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(val_int) : "memory");
    }
    __syncthreads();

    // Compute using BOTH compiler-aware and stolen (treating as 14592 floats total)
    int total_floats = 14336 + 256;
    float a = 1.0f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        int idx = (tid + i) % total_floats;
        if (idx < 256) {
            // Read from stolen
            unsigned int offset = idx * 4;
            unsigned int val;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
            a += __int_as_float((int)val);
        } else {
            a += buf[idx - 256];
        }
    }
    if (tid == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

    float *d_out;
    CK(cudaMalloc(&d_out, 4 * sm_count * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));

    // No trick: 57 KiB → 3 blocks/SM
    cudaFuncSetAttribute((void*)k_no_trick,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_USE_BYTES);
    int blocks_no_trick;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_no_trick, (void*)k_no_trick, 256, SHMEM_USE_BYTES);

    // With trick: 56 KiB declared
    cudaFuncSetAttribute((void*)k_with_trick,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 56*1024);
    int blocks_with_trick;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_with_trick, (void*)k_with_trick, 256, 56*1024);

    printf("# B300 trick perf comparison: 57 KiB usable, 256 threads/block\n");
    printf("# No trick:   declares %d B → %d blocks/SM\n",
           SHMEM_USE_BYTES, blocks_no_trick);
    printf("# With trick: declares %d B → %d blocks/SM (uses 1 KB stolen)\n\n",
           56*1024, blocks_with_trick);

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    // Test with grid that maxes out occupancy
    int n_blocks_no = blocks_no_trick * sm_count;     // = 3 × 148 = 444
    int n_blocks_yes = blocks_with_trick * sm_count;  // = 4 × 148 = 592

    float t_no_trick = bench([&]{
        k_no_trick<<<n_blocks_no, 256, SHMEM_USE_BYTES, s>>>(d_out);
    });
    float t_with_trick = bench([&]{
        k_with_trick<<<n_blocks_yes, 256, 56*1024, s>>>(d_out);
    });

    printf("# Grid sized to max blocks (no_trick=%d blocks, with_trick=%d blocks):\n", n_blocks_no, n_blocks_yes);
    printf("  No trick (3 blk/SM): %.4f ms (work units: %d × %d iter = %d Mops)\n",
           t_no_trick, n_blocks_no, ITERS, n_blocks_no * 256 * ITERS / 1000000);
    printf("  Trick (4 blk/SM):    %.4f ms (work units: %d × %d iter = %d Mops)\n",
           t_with_trick, n_blocks_yes, ITERS, n_blocks_yes * 256 * ITERS / 1000000);

    // Throughput per block
    float work_no = (float)n_blocks_no * 256 * ITERS;
    float work_yes = (float)n_blocks_yes * 256 * ITERS;
    float throughput_no = work_no / t_no_trick;
    float throughput_yes = work_yes / t_with_trick;
    printf("\n  Throughput no_trick:  %.2f Mops/ms\n", throughput_no/1e3);
    printf("  Throughput with_trick: %.2f Mops/ms (%.2fx faster!)\n",
           throughput_yes/1e3, throughput_yes/throughput_no);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
