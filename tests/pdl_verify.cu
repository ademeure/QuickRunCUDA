// Verify: is PDL "savings" real or DCE artifact?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef ITERS_K
#define ITERS_K 5000
#endif

// Style A: conditional write (impossible condition - might be DCE'd?)
extern "C" __global__ void k_pdl_A(float *out, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at, half2 = ITERS_K - signal_at;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}
extern "C" __global__ void k_pdl_first_A(float *out, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at, half2 = ITERS_K - signal_at;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}
extern "C" __global__ void k_nopdl_A(float *out) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_K; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

// Style B: unconditional write of thread 0 (definitely runs)
extern "C" __global__ void k_pdl_B(float *out, int signal_at, int k) {
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
extern "C" __global__ void k_pdl_first_B(float *out, int signal_at, int k) {
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
extern "C" __global__ void k_nopdl_B(float *out, int k) {
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

    printf("# B300 PDL anti-DCE verification: Style A vs Style B\n");
    printf("# %d blocks x %d threads, %d iters\n\n", blocks, threads, ITERS_K);

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
    CK(cudaMemset(d_out, 0, blocks * threads * sizeof(float)));

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

    int chain = 64;

    // STYLE A
    printf("## Style A (conditional write, possibly DCE'd):\n");
    {
        float t_seq = bench([&]{
            for (int k = 0; k < chain; k++)
                k_nopdl_A<<<blocks,threads,0,s>>>(d_out);
        });
        printf("  nopdl: %.4f ms (%.2f us/kernel)\n", t_seq, t_seq*1000/chain);
        for (int pct : {50, 90, 99, 100}) {
            int sig = (ITERS_K * pct) / 100;
            float t = bench([&]{
                void *fa[] = {&d_out, &sig};
                cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first_A, fa);
                for (int k = 1; k < chain; k++) {
                    void *ka[] = {&d_out, &sig};
                    cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_A, ka);
                }
            });
            printf("  pdl_%d%%: %.4f ms (save=%+.2f us/kernel)\n",
                   pct, t, (t_seq-t)*1000/chain);
        }
    }

    // STYLE B
    printf("\n## Style B (unconditional write, definitely runs):\n");
    {
        float t_seq = bench([&]{
            for (int k = 0; k < chain; k++)
                k_nopdl_B<<<blocks,threads,0,s>>>(d_out, k);
        });
        printf("  nopdl: %.4f ms (%.2f us/kernel)\n", t_seq, t_seq*1000/chain);
        for (int pct : {50, 90, 99, 100}) {
            int sig = (ITERS_K * pct) / 100;
            float t = bench([&]{
                int sk = 0;
                void *fa[] = {&d_out, &sig, &sk};
                cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first_B, fa);
                for (int k = 1; k < chain; k++) {
                    sk = k;
                    void *ka[] = {&d_out, &sig, &sk};
                    cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_B, ka);
                }
            });
            printf("  pdl_%d%%: %.4f ms (save=%+.2f us/kernel)\n",
                   pct, t, (t_seq-t)*1000/chain);
        }
    }

    CK(cudaFree(d_out));
    return 0;
}
