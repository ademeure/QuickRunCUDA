// Verify: is PDL slowdown from memory ordering or write contention?
// Test: separate output buffers per kernel
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef ITERS_K
#define ITERS_K 5000
#endif

// Separate-buffer kernels (no aliasing between consecutive kernels)
extern "C" __global__ void k_pdl_S(float *out, int signal_at, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    int half1 = signal_at, half2 = ITERS_K - signal_at;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void k_pdl_first_S(float *out, int signal_at, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    int half1 = signal_at, half2 = ITERS_K - signal_at;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void k_nopdl_S(float *out, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_K; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    printf("# B300 PDL memory aliasing test\n");
    printf("# %d blocks x %d threads, %d iters\n\n", blocks, threads, ITERS_K);

    int chain = 64;
    // Allocate one buffer per kernel — no aliasing
    std::vector<float*> bufs(chain);
    for (int i = 0; i < chain; i++) {
        CK(cudaMalloc(&bufs[i], blocks * sizeof(float)));
        CK(cudaMemset(bufs[i], 0, blocks * sizeof(float)));
    }
    // Also one shared buffer
    float *d_shared;
    CK(cudaMalloc(&d_shared, blocks * sizeof(float)));
    CK(cudaMemset(d_shared, 0, blocks * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));

    cudaLaunchAttribute attr_pdl;
    attr_pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr_pdl.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_pdl = {dim3(blocks), dim3(threads), 0, s, &attr_pdl, 1};

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

    int sig_99 = (ITERS_K * 99) / 100;

    // ========== Test A: Same buffer (aliased writes) ==========
    printf("## A: Same buffer (writes alias across kernels)\n");
    {
        float t_nopdl = bench([&]{
            for (int k = 0; k < chain; k++)
                k_nopdl_S<<<blocks,threads,0,s>>>(d_shared, k);
        });
        float t_pdl = bench([&]{
            int k0 = 0;
            void *fa[] = {&d_shared, &sig_99, &k0};
            cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first_S, fa);
            for (int k = 1; k < chain; k++) {
                int kk = k;
                void *ka[] = {&d_shared, &sig_99, &kk};
                cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_S, ka);
            }
        });
        printf("  nopdl=%.4f ms, pdl_99=%.4f ms, save=%+.2f us/kernel\n",
               t_nopdl, t_pdl, (t_nopdl-t_pdl)*1000/chain);
    }

    // ========== Test B: Separate buffers (no aliasing) ==========
    printf("\n## B: Separate buffer per kernel (no aliasing)\n");
    {
        float t_nopdl = bench([&]{
            for (int k = 0; k < chain; k++)
                k_nopdl_S<<<blocks,threads,0,s>>>(bufs[k], k);
        });
        float t_pdl = bench([&]{
            int k0 = 0;
            void *fa[] = {&bufs[0], &sig_99, &k0};
            cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first_S, fa);
            for (int k = 1; k < chain; k++) {
                int kk = k;
                void *ka[] = {&bufs[k], &sig_99, &kk};
                cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_S, ka);
            }
        });
        printf("  nopdl=%.4f ms, pdl_99=%.4f ms, save=%+.2f us/kernel\n",
               t_nopdl, t_pdl, (t_nopdl-t_pdl)*1000/chain);
    }

    // ========== Test C: NO write at all (compute-only) ==========
    printf("\n## C: NO memory writes (loop without final write)\n");
    {
        // Use the conditional-write version which never fires
        // (Loop runs but no write — measures pure PDL behavior)
        // We can simulate by setting `signal_at` such that nothing ever fires
        // Actually we need a kernel with no write — let's reuse style A
        // Just pre-compile a version
    }

    // ========== Test D: Test wait/launch overhead alone ==========
    printf("\n## D: Wait+launch_dependents alone (no useful work)\n");

    for (int i = 0; i < chain; i++) cudaFree(bufs[i]);
    cudaFree(d_shared);
    return 0;
}
