// L1 cache behavior on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Read N bytes repeatedly - small N fits in L1, large N doesn't
extern "C" __global__ void l1_test(float *src, float *dst, int N, int reps) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0;
    for (int r = 0; r < reps; r++) {
        for (int i = tid; i < N; i += stride) {
            acc += src[i];
        }
    }
    if (acc == -42.0f) dst[tid] = acc;
}

// Single SM only (1 block), measure to expose L1 behavior
extern "C" __global__ void l1_single_sm(float *src, float *dst, int N, int reps) {
    int tid = threadIdx.x;
    float acc = 0;
    for (int r = 0; r < reps; r++) {
        for (int i = tid; i < N; i += blockDim.x) {
            acc += src[i];
        }
    }
    if (acc == -42.0f) dst[tid] = acc;
}

// Use ld.global.ca (cache all = L1+L2)
extern "C" __global__ void l1_with_ca(float *src, float *dst, int N, int reps) {
    int tid = threadIdx.x;
    float acc = 0;
    for (int r = 0; r < reps; r++) {
        for (int i = tid; i < N; i += blockDim.x) {
            float v;
            asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(v) : "l"(&src[i]));
            acc += v;
        }
    }
    if (acc == -42.0f) dst[tid] = acc;
}

// Use ld.global.cg (cache global = L2 only, skip L1)
extern "C" __global__ void l1_with_cg(float *src, float *dst, int N, int reps) {
    int tid = threadIdx.x;
    float acc = 0;
    for (int r = 0; r < reps; r++) {
        for (int i = tid; i < N; i += blockDim.x) {
            float v;
            asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(&src[i]));
            acc += v;
        }
    }
    if (acc == -42.0f) dst[tid] = acc;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));

    printf("# B300 L1 cache investigation\n");
    int gl1, ll1;
    cudaDeviceGetAttribute(&gl1, cudaDevAttrGlobalL1CacheSupported, 0);
    cudaDeviceGetAttribute(&ll1, cudaDevAttrLocalL1CacheSupported, 0);
    printf("# GlobalL1CacheSupported: %d, LocalL1CacheSupported: %d\n", gl1, ll1);

    int N = 1 << 20;  // 4 MB worth of floats
    float *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, 4096 * sizeof(float));
    cudaMemset(d_src, 0x40, N * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);

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

    int reps = 100;

    // Single SM, sweep working set size
    printf("\n## Single SM read N bytes %dx (1 block × 256 thr)\n", reps);
    printf("# %-10s %-10s %-12s %-12s %-12s\n",
           "N_KB", "ratio", "default_GB/s", "ld.ca_GB/s", "ld.cg_GB/s");

    int N_kb_arr[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    for (int n_kb : N_kb_arr) {
        int n_floats = (n_kb * 1024) / 4;
        if (n_floats > N) continue;

        float t_def = bench([&]{
            l1_single_sm<<<1, 256, 0, s>>>(d_src, d_dst, n_floats, reps);
        });
        float t_ca = bench([&]{
            l1_with_ca<<<1, 256, 0, s>>>(d_src, d_dst, n_floats, reps);
        });
        float t_cg = bench([&]{
            l1_with_cg<<<1, 256, 0, s>>>(d_src, d_dst, n_floats, reps);
        });

        // Total bytes read
        size_t total_bytes = (size_t)n_floats * reps * 4;
        printf("  %-10d %-10s %-12.1f %-12.1f %-12.1f\n",
               n_kb, n_kb <= 32 ? "L1?" : (n_kb <= 79 ? "persist L2" : (n_kb <= 128 ? "L2" : "DRAM")),
               total_bytes / (t_def/1e3) / 1e9,
               total_bytes / (t_ca/1e3) / 1e9,
               total_bytes / (t_cg/1e3) / 1e9);
    }

    // Multi-SM, sweep working set
    printf("\n## Full GPU (148 blocks × 256 thr) read N bytes %dx\n", reps);
    for (int n_kb : N_kb_arr) {
        int n_floats = (n_kb * 1024) / 4;
        if (n_floats > N) continue;
        float t = bench([&]{
            l1_test<<<prop.multiProcessorCount, 256, 0, s>>>(d_src, d_dst, n_floats, reps);
        });
        size_t total = (size_t)n_floats * reps * 4 * prop.multiProcessorCount;
        printf("  N=%-6d KB: %.1f GB/s (aggregate)\n", n_kb, total / (t/1e3) / 1e9);
    }

    cudaStreamDestroy(s);
    cudaFree(d_src); cudaFree(d_dst);
    return 0;
}
