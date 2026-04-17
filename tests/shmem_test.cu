// Test B300 shared memory: 228 KB per SM, 227 KB per block opt-in
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void shmem_kernel(float *out, int iters) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    smem[tid] = tid * 1.0f;
    __syncthreads();
    float a = smem[tid];
    for (int i = 0; i < iters; i++) {
        a = smem[(tid + i) & 1023];  // shmem read pattern
    }
    if (tid == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void shmem_alloc_kernel(float *out, int iters, int alloc_bytes) {
    extern __shared__ float smem[];
    // Just ensure smem is allocated
    int tid = threadIdx.x;
    if (tid < (alloc_bytes / 4)) smem[tid] = tid;
    __syncthreads();
    float a = smem[tid % 32];
    for (int i = 0; i < iters; i++) {
        a += smem[(tid + i) & 31];
    }
    if (tid == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("# B300 shared memory test\n");
    printf("# default per-block: %zu (%.1f KB)\n", prop.sharedMemPerBlock, prop.sharedMemPerBlock/1024.f);
    printf("# opt-in per-block:  %zu (%.1f KB)\n", prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin/1024.f);
    printf("# per-SM:            %zu (%.1f KB)\n", prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor/1024.f);

    int blocks = prop.multiProcessorCount;
    int threads = 128;

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));

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

    // Test with various shmem sizes
    int sizes_kb[] = {16, 48, 96, 128, 192, 200, 220, 227};
    printf("\n# Per-block shmem allocation test\n");
    printf("# %-10s %-15s %-12s\n", "alloc_KB", "status", "time_ms");

    for (int sz : sizes_kb) {
        int sz_bytes = sz * 1024;

        // Set the dynamic shared mem opt-in if needed
        if (sz_bytes > 48 * 1024) {
            cudaFuncSetAttribute((void*)shmem_alloc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sz_bytes);
        }

        cudaError_t r;
        float t = bench([&]{
            shmem_alloc_kernel<<<blocks, threads, sz_bytes, s>>>(d_out, 100, sz_bytes);
        });
        r = cudaGetLastError();
        printf("  %-10d %-15s %.4f ms\n", sz,
               r == cudaSuccess ? "OK" : cudaGetErrorString(r), t);
    }

    // ===== Test 2: shmem random access bandwidth =====
    printf("\n# Shmem random-access bandwidth test\n");
    cudaFuncSetAttribute((void*)shmem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32 * 1024);
    int iters_arr[] = {100, 1000, 10000, 100000};
    for (int iters : iters_arr) {
        float t = bench([&]{
            shmem_kernel<<<blocks, 1024, 4096, s>>>(d_out, iters);  // 1024 floats = 4 KB
        });
        // 148 SMs × 1024 thr × N iters reads, 4-byte each
        size_t total_reads = (size_t)blocks * 1024 * iters * 4;
        float bw = total_reads / (t/1e3f) / 1e9f;
        printf("  iters=%-7d : %.4f ms = %.1f GB/s\n", iters, t, bw);
    }

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
