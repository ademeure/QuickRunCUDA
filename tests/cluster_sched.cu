// ClusterSchedulingPolicyPreference: Spread vs LoadBalancing
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void compute(float *out, int iters) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm = prop.multiProcessorCount;
    int blocks = sm, threads = 128;

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

    int iters = 5000;

    printf("# B300 ClusterSchedulingPolicy comparison\n");
    printf("# %d blocks × %d threads × %d iters\n\n", blocks, threads, iters);

    cudaClusterSchedulingPolicy policies[] = {
        cudaClusterSchedulingPolicyDefault,
        cudaClusterSchedulingPolicySpread,
        cudaClusterSchedulingPolicyLoadBalancing
    };
    const char *names[] = {"Default", "Spread", "LoadBalancing"};

    for (int csize : {2, 4, 8}) {
        printf("## Cluster size = %d\n", csize);
        int g = (sm / csize) * csize;
        for (int pi = 0; pi < 3; pi++) {
            cudaLaunchAttribute attrs[2];
            attrs[0].id = cudaLaunchAttributeClusterDimension;
            attrs[0].val.clusterDim.x = csize;
            attrs[0].val.clusterDim.y = 1;
            attrs[0].val.clusterDim.z = 1;
            attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
            attrs[1].val.clusterSchedulingPolicyPreference = policies[pi];

            cudaLaunchConfig_t cfg = {dim3(g), dim3(threads), 0, s, attrs, 2};

            float t = bench([&]{
                int it = iters;
                void *args[] = {&d_out, &it};
                cudaError_t r = cudaLaunchKernelExC(&cfg, (void*)compute, args);
                if (r != cudaSuccess) { printf("  ERR: %s\n", cudaGetErrorString(r)); return; }
            });
            printf("  %-15s : %.4f ms\n", names[pi], t);
        }
    }

    cudaFree(d_out);
    return 0;
}
