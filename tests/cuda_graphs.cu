// CUDA Graphs vs Streams comparison
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
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

    printf("# B300 CUDA Graphs vs Streams\n");
    printf("# %d blocks x %d threads\n\n", blocks, threads);

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));
    CK(cudaMemset(d_out, 0, blocks * sizeof(float)));

    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

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

    int iter_arr[] = {500, 1000, 5000, 10000};
    int chain_arr[] = {8, 32, 64, 128};

    printf("## Stream chain vs Graph chain\n");
    printf("# %-8s %-8s %-12s %-12s %-12s %-12s\n",
           "iters", "chain", "stream_us", "graph_us", "graph_save", "save/kern");

    for (int ii = 0; ii < 4; ii++) {
        int it = iter_arr[ii];
        for (int ci = 0; ci < 4; ci++) {
            int chain = chain_arr[ci];

            // Plain stream chain
            float t_stream = bench([&]{
                for (int k = 0; k < chain; k++)
                    compute<<<blocks,threads,0,s>>>(d_out, it, k);
            });

            // Capture graph
            cudaGraph_t graph;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
            for (int k = 0; k < chain; k++)
                compute<<<blocks,threads,0,s>>>(d_out, it, k);
            CK(cudaStreamEndCapture(s, &graph));

            cudaGraphExec_t graphExec;
            CK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

            // Run graph
            float t_graph = bench([&]{
                cudaGraphLaunch(graphExec, s);
            });

            float per_kern_save = (t_stream - t_graph) * 1000 / chain;
            printf("  %-8d %-8d %-12.2f %-12.2f %+-12.2f %+-12.2f\n",
                   it, chain, t_stream*1000, t_graph*1000,
                   (t_stream - t_graph)*1000, per_kern_save);

            cudaGraphExecDestroy(graphExec);
            cudaGraphDestroy(graph);
        }
    }

    // ===== Graph instantiation cost =====
    printf("\n## Graph build/instantiate/destroy cost (chain=32, 1000 iters)\n");
    {
        const int N = 100;
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        // Build cost
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < N; rep++) {
            cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
            for (int k = 0; k < 32; k++)
                compute<<<blocks,threads,0,s>>>(d_out, 1000, k);
            cudaStreamEndCapture(s, &graph);
            cudaGraphDestroy(graph);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        float build_us = std::chrono::duration<float, std::micro>(t1-t0).count() / N;
        printf("  Capture+EndCapture: %.2f us per chain\n", build_us);

        // Instantiate cost
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int k = 0; k < 32; k++) compute<<<blocks,threads,0,s>>>(d_out, 1000, k);
        cudaStreamEndCapture(s, &graph);

        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < N; rep++) {
            cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
            cudaGraphExecDestroy(graphExec);
        }
        t1 = std::chrono::high_resolution_clock::now();
        printf("  Instantiate+Destroy: %.2f us per call\n",
               std::chrono::duration<float, std::micro>(t1-t0).count() / N);

        cudaGraphDestroy(graph);
    }

    // ===== Graph upload (cudaGraphUpload) =====
    printf("\n## Graph upload + repeated execution\n");
    {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int k = 0; k < 32; k++) compute<<<blocks,threads,0,s>>>(d_out, 1000, k);
        cudaStreamEndCapture(s, &graph);
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

        float t_first = bench([&]{ cudaGraphLaunch(graphExec, s); }, 1);

        cudaGraphUpload(graphExec, s);  // pre-upload
        cudaDeviceSynchronize();
        float t_after_upload = bench([&]{ cudaGraphLaunch(graphExec, s); });

        printf("  First launch: %.2f us\n", t_first*1000);
        printf("  After upload: %.2f us\n", t_after_upload*1000);

        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    }

    CK(cudaFree(d_out));
    return 0;
}
