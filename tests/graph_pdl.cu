// CUDA Graphs + PDL combined: do the savings stack?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef ITERS_K
#define ITERS_K 1000
#endif

// Conditional-write style for compatibility
extern "C" __global__ void k_pdl(float *out, int signal_at) {
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
extern "C" __global__ void k_pdl_first(float *out, int signal_at) {
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
extern "C" __global__ void k_nopdl(float *out) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_K; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    printf("# B300 Graph+PDL combined, %d blocks × %d threads, %d iters\n\n",
           blocks, threads, ITERS_K);

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
    CK(cudaMemset(d_out, 0, blocks * threads * sizeof(float)));

    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    cudaLaunchAttribute attr_pdl;
    attr_pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr_pdl.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_pdl = {dim3(blocks), dim3(threads), 0, s, &attr_pdl, 1};

    auto bench = [&](auto fn, int trials=15) {
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

    int chain_arr[] = {8, 32, 64, 128};
    int sig_99 = (ITERS_K * 99) / 100;

    printf("# Comparison of 4 modes (chain of N kernels):\n");
    printf("# %-8s %-12s %-12s %-12s %-12s\n",
           "chain", "stream_us", "graph_us", "stream+pdl", "graph+pdl");

    for (int ci = 0; ci < 4; ci++) {
        int chain = chain_arr[ci];

        // Mode 1: Stream only (no PDL)
        float t_stream = bench([&]{
            for (int k = 0; k < chain; k++)
                k_nopdl<<<blocks,threads,0,s>>>(d_out);
        });

        // Mode 2: Graph (no PDL)
        cudaGraph_t g_nopdl;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int k = 0; k < chain; k++) k_nopdl<<<blocks,threads,0,s>>>(d_out);
        cudaStreamEndCapture(s, &g_nopdl);
        cudaGraphExec_t g_nopdl_exec;
        cudaGraphInstantiate(&g_nopdl_exec, g_nopdl, nullptr, nullptr, 0);
        float t_graph = bench([&]{ cudaGraphLaunch(g_nopdl_exec, s); });

        // Mode 3: Stream + PDL
        float t_stream_pdl = bench([&]{
            void *fa[] = {&d_out, &sig_99};
            cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first, fa);
            for (int k = 1; k < chain; k++) {
                void *ka[] = {&d_out, &sig_99};
                cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl, ka);
            }
        });

        // Mode 4: Graph + PDL
        cudaGraph_t g_pdl;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        {
            void *fa[] = {&d_out, &sig_99};
            cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first, fa);
            for (int k = 1; k < chain; k++) {
                void *ka[] = {&d_out, &sig_99};
                cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl, ka);
            }
        }
        cudaStreamEndCapture(s, &g_pdl);
        cudaGraphExec_t g_pdl_exec;
        cudaGraphInstantiate(&g_pdl_exec, g_pdl, nullptr, nullptr, 0);
        float t_graph_pdl = bench([&]{ cudaGraphLaunch(g_pdl_exec, s); });

        printf("  %-8d %-12.2f %-12.2f %-12.2f %-12.2f\n",
               chain, t_stream*1000, t_graph*1000, t_stream_pdl*1000, t_graph_pdl*1000);
        printf("           per-kernel: stream=%.2f graph=%.2f stream+pdl=%.2f graph+pdl=%.2f us\n",
               t_stream*1000/chain, t_graph*1000/chain,
               t_stream_pdl*1000/chain, t_graph_pdl*1000/chain);

        cudaGraphExecDestroy(g_nopdl_exec); cudaGraphDestroy(g_nopdl);
        cudaGraphExecDestroy(g_pdl_exec); cudaGraphDestroy(g_pdl);
    }

    CK(cudaFree(d_out));
    return 0;
}
