// PDL: find optimal signal point as function of kernel size
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void k_pdl(float *out, int total, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at;
    int half2 = total - signal_at;
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

extern "C" __global__ void k_pdl_first(float *out, int total, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at;
    int half2 = total - signal_at;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void k_nopdl(float *out, int total) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < total; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
    CK(cudaMemset(d_out, 0, blocks * threads * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    cudaLaunchAttribute attr_pdl;
    attr_pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr_pdl.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_pdl = {dim3(blocks), dim3(threads), 0, s, &attr_pdl, 1};

    auto bench = [&](auto fn, int trials=15) {
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

    int sizes[] = {500, 1000, 2500, 5000, 10000, 25000, 50000, 100000};
    int n_chain = 64;

    printf("# B300 PDL optimal signal sweep, %d blocks x %d threads, %d kernels in chain\n",
           blocks, threads, n_chain);
    printf("# Per-kernel saving = (nopdl - pdl) / n_chain (in microseconds)\n\n");

    int sig_pcts[] = {50, 75, 90, 95, 98, 99, 100};
    int n_sigs = 7;

    printf("# %-8s %-10s", "k_iters", "nopdl_us");
    for (int si = 0; si < n_sigs; si++) printf(" sig%-3d", sig_pcts[si]);
    printf("    | best_pct best_save_us\n");

    for (int sz_i = 0; sz_i < 8; sz_i++) {
        int total = sizes[sz_i];

        float t_nopdl = bench([&]{
            for (int k = 0; k < n_chain; k++)
                k_nopdl<<<blocks,threads,0,s>>>(d_out, total);
        });
        float per_kernel_nopdl_us = t_nopdl * 1000.0f / n_chain;
        printf("  %-8d %-10.2f", total, per_kernel_nopdl_us);

        float best = 1e30f;
        int best_pct = -1;
        float best_save = 0;
        for (int si = 0; si < n_sigs; si++) {
            int pct = sig_pcts[si];
            int sig = (total * pct) / 100;
            float t_pdl = bench([&]{
                void *first_args[] = {&d_out, &total, &sig};
                cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first, first_args);
                for (int k = 1; k < n_chain; k++) {
                    void *kargs[] = {&d_out, &total, &sig};
                    cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl, kargs);
                }
            });
            float save_us = (t_nopdl - t_pdl) * 1000.0f / n_chain;
            printf(" %+5.2f", save_us);
            if (t_pdl < best) { best = t_pdl; best_pct = pct; best_save = save_us; }
        }
        printf("    |   %d%%    %+.2f us\n", best_pct, best_save);
    }

    CK(cudaFree(d_out));
    return 0;
}
