// Persistent kernel pattern + PDL: how do they interact?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cooperative_groups.h>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

namespace cg = cooperative_groups;

// Persistent kernel: stays resident for N iterations
extern "C" __global__ void persistent(float *out, int n_iter, int per_iter) {
    auto grid = cg::this_grid();
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;

    for (int it = 0; it < n_iter; it++) {
        for (int i = 0; i < per_iter; i++)
            asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
        grid.sync();
    }

    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

// Persistent kernel WITH PDL signal at end of each iteration
extern "C" __global__ void persistent_pdl(float *out, int n_iter, int per_iter) {
    auto grid = cg::this_grid();
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;

    for (int it = 0; it < n_iter; it++) {
        for (int i = 0; i < per_iter; i++)
            asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
        // Signal that iteration N is "logically done" — wakes dependent kernel
        if (it == n_iter - 2) {  // signal one iter before end
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        grid.sync();
    }

    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void consumer(float *out, int per_iter) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    for (int i = 0; i < per_iter; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void consumer_nopdl(float *out, int per_iter) {
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    for (int i = 0; i < per_iter; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    printf("# B300 persistent kernel + PDL test\n");
    printf("# %d blocks × %d threads\n\n", blocks, threads);

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    cudaLaunchAttribute coop_pdl_attrs[2];
    coop_pdl_attrs[0].id = cudaLaunchAttributeCooperative;
    coop_pdl_attrs[0].val.cooperative = 1;
    coop_pdl_attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    coop_pdl_attrs[1].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_coop_pdl = {dim3(blocks), dim3(threads), 0, s, coop_pdl_attrs, 2};

    cudaLaunchAttribute coop_attr;
    coop_attr.id = cudaLaunchAttributeCooperative;
    coop_attr.val.cooperative = 1;
    cudaLaunchConfig_t cfg_coop = {dim3(blocks), dim3(threads), 0, s, &coop_attr, 1};

    cudaLaunchAttribute pdl_attr;
    pdl_attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_pdl = {dim3(blocks), dim3(threads), 0, s, &pdl_attr, 1};

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

    int n_iter = 16;
    int per_iter = 5000;

    // Method 1: persistent + consumer (sequential, no PDL)
    float t_seq = bench([&]{
        int ni = n_iter, pi = per_iter;
        void *args[] = {&d_out, &ni, &pi};
        cudaLaunchKernelExC(&cfg_coop, (void*)persistent, args);
        consumer_nopdl<<<blocks, threads, 0, s>>>(d_out, per_iter);
    });

    // Method 2: persistent_pdl + consumer (PDL signaled from persistent)
    float t_pdl = bench([&]{
        int ni = n_iter, pi = per_iter;
        void *args[] = {&d_out, &ni, &pi};
        cudaLaunchKernelExC(&cfg_coop_pdl, (void*)persistent_pdl, args);
        int cpi = per_iter;
        void *cargs[] = {&d_out, &cpi};
        cudaLaunchKernelExC(&cfg_pdl, (void*)consumer, cargs);
    });

    // Method 3: equivalent without persistent (chain of N kernels + consumer)
    float t_chain = bench([&]{
        for (int i = 0; i < n_iter; i++)
            consumer_nopdl<<<blocks, threads, 0, s>>>(d_out, per_iter);
        consumer_nopdl<<<blocks, threads, 0, s>>>(d_out, per_iter);
    });

    printf("# Equivalent: %d iters × %d FFMA + 1 consumer (%d FFMA)\n", n_iter, per_iter, per_iter);
    printf("  Persistent + consumer (no PDL):       %.3f ms\n", t_seq);
    printf("  Persistent_PDL + consumer (PDL):      %.3f ms (save %+.3f)\n",
           t_pdl, t_seq - t_pdl);
    printf("  Kernel chain equivalent:              %.3f ms\n", t_chain);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
