// Performance: does the 'steal reserved' trick HURT throughput?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define ITERS 1000

// Plain 56 KiB shmem kernel
extern "C" __global__ void k_56k_plain(float *out) {
    extern __shared__ float buf[];
    int tid = threadIdx.x;
    // Use shmem in compute loop
    if (tid < 1024) buf[tid] = (float)tid;
    __syncthreads();

    float a = 1.0f + tid * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a += buf[(tid + i) & 1023];
    }
    if (tid == 0) out[blockIdx.x] = a;
}

// 56 KiB + steal 1 KiB (writes to offset 0..1023)
extern "C" __global__ void k_57k_stealing(float *out) {
    extern __shared__ float buf[];
    int tid = threadIdx.x;
    if (tid < 1024) buf[tid] = (float)tid;

    // Steal: also write reserved
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val = i + 0x100;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(val) : "memory");
    }
    __syncthreads();

    float a = 1.0f + tid * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a += buf[(tid + i) & 1023];
        // Also touch stolen area
        if (i % 4 == 0) {
            unsigned int offset = (i & 0xFC);
            unsigned int val;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
            a += __int_as_float((int)val);
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

    cudaFuncSetAttribute((void*)k_56k_plain,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 56*1024);
    cudaFuncSetAttribute((void*)k_57k_stealing,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 56*1024);

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

    int n_blocks = 4 * sm_count;
    int n_threads = 1024;

    printf("# B300 'steal reserved' performance test\n");
    printf("# 4 blocks/SM × %d SMs = %d blocks, %d threads each, %d iter compute\n\n",
           sm_count, n_blocks, n_threads, ITERS);

    float t_plain = bench([&]{
        k_56k_plain<<<n_blocks, n_threads, 56*1024, s>>>(d_out);
    });
    float t_steal = bench([&]{
        k_57k_stealing<<<n_blocks, n_threads, 56*1024, s>>>(d_out);
    });

    printf("  56 KiB plain (no steal):       %.4f ms\n", t_plain);
    printf("  56 KiB + 1 KiB stolen:         %.4f ms (overhead %+.4f ms = %+.1f%%)\n",
           t_steal, t_steal - t_plain, (t_steal - t_plain) / t_plain * 100);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
