// Host callbacks: cudaLaunchHostFunc vs cudaStreamAddCallback
// Plus stream capture modes
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <atomic>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void quick(float *out) {
    if (threadIdx.x == 0) out[blockIdx.x] = 1.0f;
}

std::atomic<int> cb_counter{0};

void CUDART_CB host_fn(void *user_data) {
    cb_counter.fetch_add(1, std::memory_order_relaxed);
}

void CUDART_CB legacy_callback(cudaStream_t s, cudaError_t status, void *user_data) {
    cb_counter.fetch_add(1, std::memory_order_relaxed);
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        cb_counter = 0;
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

    // ===== Single host callback overhead =====
    printf("# Host callback overhead (single):\n");
    {
        float t_kern = bench([&]{ quick<<<blocks,128,0,s>>>(d_out); });
        float t_host = bench([&]{ cudaLaunchHostFunc(s, host_fn, nullptr); });
        float t_legacy = bench([&]{ cudaStreamAddCallback(s, legacy_callback, nullptr, 0); });

        printf("  kernel only:                  %.2f us\n", t_kern*1000);
        printf("  cudaLaunchHostFunc only:      %.2f us\n", t_host*1000);
        printf("  cudaStreamAddCallback only:   %.2f us\n", t_legacy*1000);
    }

    // ===== N callbacks in chain =====
    printf("\n# N host callbacks back to back\n");
    for (int N : {1, 8, 64, 256, 1000}) {
        float t1 = bench([&]{
            for (int i = 0; i < N; i++) cudaLaunchHostFunc(s, host_fn, nullptr);
        });
        float t2 = bench([&]{
            for (int i = 0; i < N; i++) cudaStreamAddCallback(s, legacy_callback, nullptr, 0);
        });
        printf("  N=%-4d : LaunchHostFunc %.2f us (%.3f us/cb), AddCallback %.2f us (%.3f us/cb)\n",
               N, t1*1000, t1*1000/N, t2*1000, t2*1000/N);
    }

    // ===== Capture cost with vs without callbacks =====
    printf("\n# Stream capture cost\n");
    {
        cudaGraph_t g;
        const int N = 32;

        // Just kernels
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 100; rep++) {
            cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
            for (int i = 0; i < N; i++) quick<<<blocks,128,0,s>>>(d_out);
            cudaStreamEndCapture(s, &g);
            cudaGraphDestroy(g);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("  capture %d kernels:    %.2f us\n", N,
               std::chrono::duration<float,std::micro>(t1-t0).count()/100);

        // Mix with callbacks
        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 100; rep++) {
            cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
            for (int i = 0; i < N; i++) {
                quick<<<blocks,128,0,s>>>(d_out);
                cudaLaunchHostFunc(s, host_fn, nullptr);
            }
            cudaStreamEndCapture(s, &g);
            cudaGraphDestroy(g);
        }
        t1 = std::chrono::high_resolution_clock::now();
        printf("  capture %d kernels+CBs: %.2f us\n", N,
               std::chrono::duration<float,std::micro>(t1-t0).count()/100);
    }

    cudaFree(d_out);
    cudaStreamDestroy(s);
    return 0;
}
