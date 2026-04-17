// Dynamic parallelism + PDL test: child kernel launch from device
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void child(float *out, int iters) {
    float a = 1.0f + threadIdx.x * 0.001f;
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void parent_dyn(float *out, int n_children, int child_iters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n_children; i++) {
            child<<<32, 128>>>(out, child_iters);
        }
    }
}

extern "C" __global__ void parent_dyn_pdl(float *out, int n_children, int child_iters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n_children; i++) {
            // Dynamic parallelism child launch — can it use PDL?
            // It's just nested launch; no special PDL attr from device
            child<<<32, 128>>>(out, child_iters);
        }
    }
    // Parent could signal launch_dependents
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));

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

    int n_children = 8;
    int child_iters = 5000;

    // Method 1: Host launches child kernels in chain
    float t_host = bench([&]{
        for (int i = 0; i < n_children; i++)
            child<<<32, 128, 0, s>>>(d_out, child_iters);
    });

    // Method 2: Dynamic parallelism (one parent, many children)
    float t_dyn = bench([&]{
        parent_dyn<<<1, 1, 0, s>>>(d_out, n_children, child_iters);
    });

    printf("# B300 Dynamic Parallelism vs Host Launches\n");
    printf("# %d child kernels × %d iters\n\n", n_children, child_iters);
    printf("  Host-side chain:    %.4f ms\n", t_host);
    printf("  Dynamic parallel:   %.4f ms (diff %+.4f)\n", t_dyn, t_dyn - t_host);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
