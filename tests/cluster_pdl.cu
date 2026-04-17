// Cluster (CGA) dimension + PDL combined
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Cluster kernel — uses __cluster_dims__ to declare cluster size
extern "C" __global__ void __cluster_dims__(2, 1, 1) cluster_kernel(
    float *out, int iters, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void plain_kernel(float *out, int iters, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

    int max_cluster, cluster_supported;
    cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, 0);
    printf("# B300 Cluster Launch supported: %d\n", cluster_supported);

    int blocks = sm_count, threads = 128;

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));
    CK(cudaMemset(d_out, 0, blocks * sizeof(float)));

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

    // ===== Test 1: Plain vs cluster (declared via attribute) =====
    printf("\n## Cluster sizes\n");

    float t_plain = bench([&]{
        plain_kernel<<<blocks,threads,0,s>>>(d_out, iters, 0);
    });
    printf("  plain (no cluster):   %.4f ms\n", t_plain);

    // Cluster via __cluster_dims__ (size=2)
    float t_cluster_static = bench([&]{
        cluster_kernel<<<blocks,threads,0,s>>>(d_out, iters, 0);
    });
    printf("  __cluster_dims__(2):  %.4f ms (ratio %.2fx)\n", t_cluster_static, t_cluster_static / t_plain);

    // ===== Test 2: Cluster via launch attribute (dynamic) =====
    printf("\n## Cluster via cudaLaunchAttributeClusterDimension\n");
    for (int csize : {1, 2, 4, 8}) {
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = csize;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;

        // gridDim must be a multiple of cluster
        int g = (sm_count / csize) * csize;
        cudaLaunchConfig_t cfg = {dim3(g), dim3(threads), 0, s, &attr, 1};

        float t = bench([&]{
            int it = iters, k = 0;
            void *args[] = {&d_out, &it, &k};
            cudaError_t r = cudaLaunchKernelExC(&cfg, (void*)plain_kernel, args);
            if (r != cudaSuccess) printf("  ERROR: %s\n", cudaGetErrorString(r));
        });
        printf("  cluster_dim=%-2d: %.4f ms (gridDim=%d)\n", csize, t, g);
    }

    // ===== Test 3: Cluster + PDL combined =====
    printf("\n## Cluster + PDL\n");
    {
        cudaEvent_t pdl_evt;
        cudaEventCreateWithFlags(&pdl_evt, cudaEventDisableTiming);

        cudaStream_t s2;
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

        int csize = 2;
        cudaLaunchAttribute attrs[2];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = csize;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        attrs[1].id = cudaLaunchAttributeProgrammaticEvent;
        attrs[1].val.programmaticEvent.event = pdl_evt;
        attrs[1].val.programmaticEvent.flags = 0;
        attrs[1].val.programmaticEvent.triggerAtBlockStart = 0;

        int g = (sm_count / csize) * csize;
        cudaLaunchConfig_t cfg_prod = {dim3(g), dim3(threads), 0, s, attrs, 2};

        // Sequential baseline (cluster dim, no PDL)
        float t_seq = bench([&]{
            cluster_kernel<<<g,threads,0,s>>>(d_out, iters, 0);
            cluster_kernel<<<g,threads,0,s>>>(d_out, iters, 1);
        });

        float t_pdl_cluster = bench([&]{
            int it = iters, k = 0;
            void *args[] = {&d_out, &it, &k};
            cudaLaunchKernelExC(&cfg_prod, (void*)plain_kernel, args);
            cudaStreamWaitEvent(s2, pdl_evt, cudaEventWaitDefault);
            cluster_kernel<<<g,threads,0,s2>>>(d_out, iters, 1);
        });

        printf("  seq cluster pair:       %.4f ms\n", t_seq);
        printf("  PDL cluster cross-stream: %.4f ms (save %+.4f)\n",
               t_pdl_cluster, t_seq - t_pdl_cluster);

        cudaStreamDestroy(s2);
        cudaEventDestroy(pdl_evt);
    }

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
