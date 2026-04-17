// PDL chain test: many short kernels back-to-back with PDL
// This is the LLM-serving scenario: deep pipelines of short kernels
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef ITERS_K
#define ITERS_K 5000
#endif

// Kernel that signals at 90% to allow next launch to start early
extern "C" __global__ void k_pdl(float *out, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at;
    int half2 = ITERS_K - signal_at;
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

extern "C" __global__ void k_nopdl(float *out) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_K; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

// First kernel in chain (no wait) - signals like the others
extern "C" __global__ void k_pdl_first(float *out, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at;
    int half2 = ITERS_K - signal_at;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

int main(int argc, char **argv) {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;
    int blocks = sm_count, threads = 128;

    printf("# B300 PDL chain: %d blocks x %d threads, ITERS_K=%d\n",
           blocks, threads, ITERS_K);

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
    CK(cudaMemset(d_out, 0, blocks * threads * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    cudaLaunchAttribute attr_pdl;
    attr_pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr_pdl.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_pdl = {dim3(blocks), dim3(threads), 0, s, &attr_pdl, 1};
    cudaLaunchConfig_t cfg_plain = {dim3(blocks), dim3(threads), 0, s, nullptr, 0};

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 3; i++) fn();
        CK(cudaStreamSynchronize(s));
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            CK(cudaEventRecord(e0, s));
            fn();
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms < best) best = ms;
        }
        return best;
    };

    // ===== Chain length sweep =====
    int chain_lens[] = {1, 2, 4, 8, 16, 32, 64, 128};

    printf("\n## Chain of %d-iter kernels: NO PDL vs PDL\n", ITERS_K);
    printf("# %-8s %-12s %-12s %-12s %-12s %-12s\n",
           "n_kerns", "nopdl_ms", "pdl_50_ms", "pdl_25_ms", "pdl_75_ms", "pdl_95_ms");

    for (int ci = 0; ci < 8; ci++) {
        int n = chain_lens[ci];

        // Sequential (no PDL)
        float t_seq = bench([&]{
            for (int k = 0; k < n; k++)
                k_nopdl<<<blocks,threads,0,s>>>(d_out);
        });

        int sigs[] = {ITERS_K/2, ITERS_K/4, 3*ITERS_K/4, 19*ITERS_K/20};
        float pdl_times[4];
        for (int si = 0; si < 4; si++) {
            int sig = sigs[si];
            pdl_times[si] = bench([&]{
                void *first_args[] = {&d_out, &sig};
                cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first, first_args);
                for (int k = 1; k < n; k++) {
                    void *kargs[] = {&d_out, &sig};
                    cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl, kargs);
                }
            });
        }
        printf("  %-8d %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n",
               n, t_seq, pdl_times[0], pdl_times[1], pdl_times[2], pdl_times[3]);
    }

    CK(cudaFree(d_out));
    return 0;
}
