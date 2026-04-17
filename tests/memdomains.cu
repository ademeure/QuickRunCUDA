// MemSyncDomain test: B300 has 4 domains
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void compute(float *out, int iters, int k) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + k * 0.00001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

// Memory writer (atomic) — for testing fence cost across domains
extern "C" __global__ void atomic_writer(unsigned int *counter, int iters) {
    for (int i = 0; i < iters; i++)
        atomicAdd(counter, 1);
}

extern "C" __global__ void mem_dependent(float *src, float *dst, int n) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] = src[tid] * 2.0f;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    int domain_count;
    CK(cudaDeviceGetAttribute(&domain_count, cudaDevAttrMemSyncDomainCount, 0));
    printf("# B300 MemSyncDomainCount: %d\n", domain_count);
    printf("# %d blocks × %d threads\n\n", blocks, threads);

    float *d_a, *d_b, *d_c;
    int N = 1 << 20;
    CK(cudaMalloc(&d_a, N * sizeof(float)));
    CK(cudaMalloc(&d_b, N * sizeof(float)));
    CK(cudaMalloc(&d_c, N * sizeof(float)));
    CK(cudaMemset(d_a, 0x40, N * sizeof(float)));

    cudaStream_t s0, s1, s2;
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

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

    // ===== Test 1: Per-stream MemSyncDomain =====
    printf("## Test 1: Stream-attached memSyncDomain\n");
    {
        // Set MemSyncDomain on streams
        cudaLaunchAttributeValue val0, val1;
        val0.memSyncDomain = cudaLaunchMemSyncDomainDefault;
        val1.memSyncDomain = cudaLaunchMemSyncDomainRemote;

        CK(cudaStreamSetAttribute(s0, cudaLaunchAttributeMemSyncDomain, &val0));
        CK(cudaStreamSetAttribute(s1, cudaLaunchAttributeMemSyncDomain, &val1));

        // Two streams launching sequentially
        float t_default = bench([&]{
            compute<<<blocks,threads,0,s0>>>(d_a, 5000, 0);
            compute<<<blocks,threads,0,s1>>>(d_b, 5000, 1);
        });
        printf("  default+remote domains parallel: %.4f ms\n", t_default);

        // Reset to default
        val0.memSyncDomain = cudaLaunchMemSyncDomainDefault;
        cudaStreamSetAttribute(s0, cudaLaunchAttributeMemSyncDomain, &val0);
        val1.memSyncDomain = cudaLaunchMemSyncDomainDefault;
        cudaStreamSetAttribute(s1, cudaLaunchAttributeMemSyncDomain, &val1);

        float t_same = bench([&]{
            compute<<<blocks,threads,0,s0>>>(d_a, 5000, 0);
            compute<<<blocks,threads,0,s1>>>(d_b, 5000, 1);
        });
        printf("  both default domain parallel:    %.4f ms\n", t_same);
    }

    // ===== Test 2: MemSyncDomain explanation =====
    printf("\n## Test 2: MemSyncDomain semantics (B300 has %d domains)\n", domain_count);
    printf("  Same-domain kernels: GPU-scope sync sufficient (cheaper fences)\n");
    printf("  Cross-domain kernels: SYSTEM-scope sync required (more expensive)\n");
    printf("  Default values: cudaLaunchMemSyncDomainDefault=0, cudaLaunchMemSyncDomainRemote=1\n");

    // ===== Test 3: PDL ProgrammaticStreamSerialization across MemSyncDomains =====
    printf("\n## Test 3: PDL across different MemSyncDomains\n");
    {
        // Set s0 = domain 0, s1 = domain 1
        cudaLaunchAttributeValue v;
        v.memSyncDomain = cudaLaunchMemSyncDomainDefault;
        cudaStreamSetAttribute(s0, cudaLaunchAttributeMemSyncDomain, &v);
        v.memSyncDomain = cudaLaunchMemSyncDomainRemote;
        cudaStreamSetAttribute(s1, cudaLaunchAttributeMemSyncDomain, &v);

        cudaEvent_t pdl_evt;
        CK(cudaEventCreateWithFlags(&pdl_evt, cudaEventDisableTiming));

        cudaLaunchAttribute prod_attrs[1];
        prod_attrs[0].id = cudaLaunchAttributeProgrammaticEvent;
        prod_attrs[0].val.programmaticEvent.event = pdl_evt;
        prod_attrs[0].val.programmaticEvent.flags = 0;
        prod_attrs[0].val.programmaticEvent.triggerAtBlockStart = 1;
        cudaLaunchConfig_t cfg = {dim3(blocks), dim3(threads), 0, s0, prod_attrs, 1};

        float t_cross_domain = bench([&]{
            int it = 5000, k = 0;
            void *args[] = {&d_a, &it, &k};
            cudaLaunchKernelExC(&cfg, (void*)compute, args);
            cudaStreamWaitEvent(s1, pdl_evt, cudaEventWaitDefault);
            compute<<<blocks,threads,0,s1>>>(d_b, 5000, 1);
        });
        printf("  PDL cross-domain s0(default) → s1(remote): %.4f ms\n", t_cross_domain);

        // Same domain
        v.memSyncDomain = cudaLaunchMemSyncDomainDefault;
        cudaStreamSetAttribute(s1, cudaLaunchAttributeMemSyncDomain, &v);

        float t_same_domain = bench([&]{
            int it = 5000, k = 0;
            void *args[] = {&d_a, &it, &k};
            cudaLaunchKernelExC(&cfg, (void*)compute, args);
            cudaStreamWaitEvent(s1, pdl_evt, cudaEventWaitDefault);
            compute<<<blocks,threads,0,s1>>>(d_b, 5000, 1);
        });
        printf("  PDL same-domain  s0(default) → s1(default): %.4f ms\n", t_same_domain);

        cudaEventDestroy(pdl_evt);
    }

    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
