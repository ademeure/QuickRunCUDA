// Per-thread default stream comparison
// Compile with -default-stream legacy (default) vs per-thread
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <vector>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void compute(float *out, int iters, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    printf("# Built with PER_THREAD_DEFAULT_STREAM\n");
#else
    printf("# Built with LEGACY default stream (NULL stream)\n");
#endif

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));
    CK(cudaMemset(d_out, 0, blocks * sizeof(float)));

    int iters = 50000;

    // Single thread, NULL stream + explicit stream concurrency
    cudaStream_t s_explicit;
    CK(cudaStreamCreateWithFlags(&s_explicit, cudaStreamNonBlocking));

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

    float t_single = bench([&]{ compute<<<blocks,threads,0,0>>>(d_out, iters, 0); });

    // Default stream + explicit stream concurrency
    float t_pair = bench([&]{
        compute<<<blocks,threads,0,0>>>(d_out, iters, 0);
        compute<<<blocks,threads,0,s_explicit>>>(d_out, iters, 1);
    });
    printf("# default + explicit nonblocking stream:\n");
    printf("# single = %.4f ms\n", t_single);
    printf("# pair   = %.4f ms (legacy: ~1×, PTDS: ~1×)\n", t_pair);

    // Multi-thread NULL stream issues from different host threads
    printf("\n# Multi-thread NULL stream concurrency (4 host threads each launch on NULL):\n");
    {
        const int nthreads = 4;
        std::vector<std::thread> threads_v;
        // Warmup
        compute<<<blocks,threads,0,0>>>(d_out, iters, 0);
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < nthreads; t++) {
            threads_v.emplace_back([&, t](){
                cudaSetDevice(0);
                compute<<<blocks,threads,0,0>>>(d_out, iters, t);
                cudaDeviceSynchronize();
            });
        }
        for (auto &th : threads_v) th.join();
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
        printf("  4-thread NULL stream parallel: %.4f ms\n", ms);
        printf("  (legacy: serializes ≈ %.1f ms; PTDS: parallel ≈ %.1f ms)\n",
               nthreads*t_single, t_single);
    }

    cudaStreamDestroy(s_explicit);
    cudaFree(d_out);
    return 0;
}
