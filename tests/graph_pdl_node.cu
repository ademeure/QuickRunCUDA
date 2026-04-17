// PDL + CUDA Graph nodes: explicit graph node API with PDL attribute
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef ITERS_K
#define ITERS_K 5000
#endif

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

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

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

    int sig = (ITERS_K * 99) / 100;
    int chain = 32;

    // ===== Build graph EXPLICITLY (not via stream capture) with PDL nodes =====
    cudaGraph_t graph;
    CK(cudaGraphCreate(&graph, 0));

    cudaGraphNode_t prev = nullptr;
    cudaKernelNodeParams kparams = {};
    kparams.gridDim = dim3(blocks);
    kparams.blockDim = dim3(threads);
    kparams.sharedMemBytes = 0;
    kparams.extra = nullptr;

    for (int k = 0; k < chain; k++) {
        kparams.func = (k == 0) ? (void*)k_pdl_first : (void*)k_pdl;
        void *args[] = {&d_out, &sig};
        kparams.kernelParams = args;

        cudaGraphNode_t node;
        cudaGraphNode_t deps[1] = {prev};
        if (k == 0) {
            CK(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &kparams));
        } else {
            CK(cudaGraphAddKernelNode(&node, graph, deps, 1, &kparams));
        }

        // Set PDL attribute on the node
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cudaError_t r = cudaGraphKernelNodeSetAttribute(node, cudaLaunchAttributeProgrammaticStreamSerialization, &attr.val);
        if (r != cudaSuccess) {
            printf("  set attr: %s\n", cudaGetErrorString(r));
        }

        prev = node;
    }
    cudaGraphExec_t graph_exec;
    CK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // Comparison: stream chain (no PDL) baseline
    float t_stream_chain = bench([&]{
        for (int k = 0; k < chain; k++)
            k_nopdl<<<blocks, threads, 0, s>>>(d_out);
    });

    // Stream chain + PDL
    cudaLaunchAttribute attr_pdl;
    attr_pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr_pdl.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_pdl = {dim3(blocks), dim3(threads), 0, s, &attr_pdl, 1};

    float t_stream_pdl = bench([&]{
        void *fa[] = {&d_out, &sig};
        cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first, fa);
        for (int k = 1; k < chain; k++) {
            void *ka[] = {&d_out, &sig};
            cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl, ka);
        }
    });

    // Graph (captured, no PDL)
    cudaGraph_t g_cap;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int k = 0; k < chain; k++) k_nopdl<<<blocks, threads, 0, s>>>(d_out);
    cudaStreamEndCapture(s, &g_cap);
    cudaGraphExec_t g_cap_exec;
    cudaGraphInstantiate(&g_cap_exec, g_cap, nullptr, nullptr, 0);
    float t_graph_capt = bench([&]{ cudaGraphLaunch(g_cap_exec, s); });

    // Graph captured WITH PDL
    cudaGraph_t g_cap_pdl;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    {
        void *fa[] = {&d_out, &sig};
        cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first, fa);
        for (int k = 1; k < chain; k++) {
            void *ka[] = {&d_out, &sig};
            cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl, ka);
        }
    }
    cudaStreamEndCapture(s, &g_cap_pdl);
    cudaGraphExec_t g_cap_pdl_exec;
    cudaGraphInstantiate(&g_cap_pdl_exec, g_cap_pdl, nullptr, nullptr, 0);
    float t_graph_cap_pdl = bench([&]{ cudaGraphLaunch(g_cap_pdl_exec, s); });

    // Graph EXPLICITLY built with PDL on each node
    float t_graph_exp_pdl = bench([&]{ cudaGraphLaunch(graph_exec, s); });

    printf("# PDL × Graph variants, %d-kernel chain, %d iter/kernel\n\n", chain, ITERS_K);
    printf("  stream (no PDL):                    %.4f ms\n", t_stream_chain);
    printf("  stream + PDL@99 (cudaLaunchKEx):    %.4f ms (save %+.4f)\n",
           t_stream_pdl, t_stream_chain - t_stream_pdl);
    printf("  graph from capture (no PDL):        %.4f ms (save %+.4f)\n",
           t_graph_capt, t_stream_chain - t_graph_capt);
    printf("  graph from capture WITH PDL@99:     %.4f ms (save %+.4f)\n",
           t_graph_cap_pdl, t_stream_chain - t_graph_cap_pdl);
    printf("  graph EXPLICIT, PDL attr per node:  %.4f ms (save %+.4f)\n",
           t_graph_exp_pdl, t_stream_chain - t_graph_exp_pdl);

    cudaGraphExecDestroy(graph_exec); cudaGraphDestroy(graph);
    cudaGraphExecDestroy(g_cap_exec); cudaGraphDestroy(g_cap);
    cudaGraphExecDestroy(g_cap_pdl_exec); cudaGraphDestroy(g_cap_pdl);
    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
