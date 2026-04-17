// Measure pure griddepcontrol overhead in isolation
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// Empty kernels
extern "C" __global__ void k_empty() {}
extern "C" __global__ void k_empty_write(float *out) {
    if (threadIdx.x == 0) out[blockIdx.x] = 0.0f;
}

// PDL primitives in isolation
extern "C" __global__ void k_signal_only() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
extern "C" __global__ void k_wait_only() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}
extern "C" __global__ void k_signal_then_wait() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

// Tiny FFMA loop to measure consumer-side wait
extern "C" __global__ void k_tiny_compute(float *out, int iters) {
    float a = 1.0f + threadIdx.x * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void k_tiny_compute_wait(float *out, int iters) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float a = 1.0f + threadIdx.x * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

    printf("# B300 PDL primitive overhead measurement\n\n");

    float *d_out;
    CK(cudaMalloc(&d_out, 1024 * sizeof(float)));
    CK(cudaMemset(d_out, 0, 1024 * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));

    cudaLaunchAttribute attr_pdl;
    attr_pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr_pdl.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_pdl_148_128 = {dim3(sm_count), dim3(128), 0, s, &attr_pdl, 1};
    cudaLaunchConfig_t cfg_pdl_1_32 = {dim3(1), dim3(32), 0, s, &attr_pdl, 1};

    auto bench = [&](auto fn, int trials=20) {
        for (int i = 0; i < 3; i++) { fn(); cudaDeviceSynchronize(); }
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

    // ===== Single kernel times =====
    printf("## Single-kernel times (us):\n");
    printf("  empty (1 block, 32 threads):           %6.2f us\n", bench([&]{ k_empty<<<1,32,0,s>>>(); })*1000);
    printf("  empty (148 blocks, 128 threads):       %6.2f us\n", bench([&]{ k_empty<<<sm_count,128,0,s>>>(); })*1000);
    printf("  signal_only (1 block, 32 threads):     %6.2f us\n", bench([&]{ k_signal_only<<<1,32,0,s>>>(); })*1000);
    printf("  signal_only (148 blocks, 128 threads): %6.2f us\n", bench([&]{ k_signal_only<<<sm_count,128,0,s>>>(); })*1000);
    printf("  wait_only (148 blocks, 128 threads):   %6.2f us\n", bench([&]{ k_wait_only<<<sm_count,128,0,s>>>(); })*1000);
    printf("  signal+wait (148 blocks):              %6.2f us\n", bench([&]{ k_signal_then_wait<<<sm_count,128,0,s>>>(); })*1000);

    // ===== Pair: PDL signal + wait kernel =====
    printf("\n## PDL pairs (1 block × 32 threads):\n");
    {
        // Sequential: empty + empty
        float t_seq = bench([&]{
            k_empty<<<1,32,0,s>>>();
            k_empty<<<1,32,0,s>>>();
        });
        printf("  empty + empty (no PDL):             %.2f us\n", t_seq*1000);
        float t_pdl = bench([&]{
            cudaLaunchKernelExC(&cfg_pdl_1_32, (void*)k_signal_only, nullptr);
            k_wait_only<<<1,32,0,s>>>();
        });
        printf("  signal + wait (PDL):                %.2f us (save %+.2f us)\n",
               t_pdl*1000, (t_seq-t_pdl)*1000);
    }

    // ===== Pair: PDL signal + wait kernel (148 blocks) =====
    printf("\n## PDL pairs (148 blocks × 128 threads):\n");
    {
        float t_seq = bench([&]{
            k_empty<<<sm_count,128,0,s>>>();
            k_empty<<<sm_count,128,0,s>>>();
        });
        printf("  empty + empty (no PDL):             %.2f us\n", t_seq*1000);

        float t_pdl = bench([&]{
            cudaLaunchKernelExC(&cfg_pdl_148_128, (void*)k_signal_only, nullptr);
            k_wait_only<<<sm_count,128,0,s>>>();
        });
        printf("  signal + wait (PDL):                %.2f us (save %+.2f us)\n",
               t_pdl*1000, (t_seq-t_pdl)*1000);

        // What if both have signal+wait?
        float t_pdl_both = bench([&]{
            cudaLaunchKernelExC(&cfg_pdl_148_128, (void*)k_signal_then_wait, nullptr);
            k_signal_then_wait<<<sm_count,128,0,s>>>();
        });
        printf("  signal+wait + signal+wait (PDL):    %.2f us (save %+.2f us)\n",
               t_pdl_both*1000, (t_seq-t_pdl_both)*1000);
    }

    // ===== Tiny compute pair, vary consumer iters =====
    printf("\n## Pair: producer signal_only + consumer with N iter wait+compute (148 blocks × 128)\n");
    int iters_arr[] = {0, 100, 500, 1000, 5000, 10000};
    for (int ii = 0; ii < 6; ii++) {
        int iters = iters_arr[ii];
        float t_seq = bench([&]{
            k_empty<<<sm_count,128,0,s>>>();
            k_tiny_compute<<<sm_count,128,0,s>>>(d_out, iters);
        });
        float t_pdl = bench([&]{
            cudaLaunchKernelExC(&cfg_pdl_148_128, (void*)k_signal_only, nullptr);
            int it = iters;
            void *args[] = {&d_out, &it};
            cudaLaunchKernelExC(&cfg_pdl_148_128, (void*)k_tiny_compute_wait, args);
        });
        printf("  iters=%-6d : seq=%.2f us, pdl=%.2f us, save %+.2f us\n",
               iters, t_seq*1000, t_pdl*1000, (t_seq-t_pdl)*1000);
    }

    CK(cudaFree(d_out));
    return 0;
}
