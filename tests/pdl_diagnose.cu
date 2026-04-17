// PDL diagnostic: isolate what causes the slowdown
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef ITERS_P
#define ITERS_P 200000
#endif
#ifndef ITERS_C
#define ITERS_C 100000
#endif

// 4 producer variants
extern "C" __global__ void p_nopdl(float *out) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_P; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void p_signal_early(float *out) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_P; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void p_signal_late(float *out) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_P; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

// 4 consumer variants
extern "C" __global__ void c_nopdl(float *out) {
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_C; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0002f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void c_with_wait(float *out) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_C; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0002f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void c_no_wait_no_work(float *out) {
    if (threadIdx.x == 0) out[blockIdx.x] = 0.0f;
}

extern "C" __global__ void c_with_wait_no_work(float *out) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (threadIdx.x == 0) out[blockIdx.x] = 0.0f;
}

int main(int argc, char **argv) {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

    int blocks = sm_count;
    int threads = 128;
    if (argc > 1) blocks = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    printf("# B300 PDL diagnostic: %d blocks x %d threads, ITERS_P=%d, ITERS_C=%d\n",
           blocks, threads, ITERS_P, ITERS_C);

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

    void *out_args[] = {&d_out};

    // ===== Standalone kernel times =====
    printf("\n## Standalone kernel times\n");
    float t = bench([&]{ p_nopdl<<<blocks,threads,0,s>>>(d_out); });
    printf("p_nopdl:        %.4f ms\n", t);
    t = bench([&]{ cudaLaunchKernelExC(&cfg_pdl, (void*)p_signal_early, out_args); });
    printf("p_signal_early: %.4f ms (PDL launch attr)\n", t);
    t = bench([&]{ cudaLaunchKernelExC(&cfg_pdl, (void*)p_signal_late, out_args); });
    printf("p_signal_late:  %.4f ms (PDL launch attr)\n", t);
    t = bench([&]{ c_nopdl<<<blocks,threads,0,s>>>(d_out); });
    printf("c_nopdl:        %.4f ms\n", t);
    t = bench([&]{ c_with_wait<<<blocks,threads,0,s>>>(d_out); });
    printf("c_with_wait (no producer):  %.4f ms (wait will return immediately)\n", t);

    // ===== Pair tests =====
    printf("\n## Pair tests (producer + consumer)\n");

    // Sequential baseline
    t = bench([&]{
        p_nopdl<<<blocks,threads,0,s>>>(d_out);
        c_nopdl<<<blocks,threads,0,s>>>(d_out);
    });
    printf("Seq (p_nopdl + c_nopdl):                 %.4f ms\n", t);

    // PDL on producer only (consumer no wait)
    t = bench([&]{
        cudaLaunchKernelExC(&cfg_pdl, (void*)p_signal_early, out_args);
        c_nopdl<<<blocks,threads,0,s>>>(d_out);
    });
    printf("PDL (p_signal_early + c_nopdl):          %.4f ms (no consumer wait)\n", t);

    t = bench([&]{
        cudaLaunchKernelExC(&cfg_pdl, (void*)p_signal_late, out_args);
        c_nopdl<<<blocks,threads,0,s>>>(d_out);
    });
    printf("PDL (p_signal_late + c_nopdl):           %.4f ms (no consumer wait)\n", t);

    // PDL on producer + consumer with wait
    t = bench([&]{
        cudaLaunchKernelExC(&cfg_pdl, (void*)p_signal_early, out_args);
        cudaLaunchKernelExC(&cfg_plain, (void*)c_with_wait, out_args);
    });
    printf("PDL (p_signal_early + c_with_wait):      %.4f ms\n", t);

    t = bench([&]{
        cudaLaunchKernelExC(&cfg_pdl, (void*)p_signal_late, out_args);
        cudaLaunchKernelExC(&cfg_plain, (void*)c_with_wait, out_args);
    });
    printf("PDL (p_signal_late + c_with_wait):       %.4f ms\n", t);

    // PDL with empty consumer
    t = bench([&]{
        cudaLaunchKernelExC(&cfg_pdl, (void*)p_signal_early, out_args);
        c_no_wait_no_work<<<blocks,threads,0,s>>>(d_out);
    });
    printf("PDL (p_signal_early + EMPTY consumer):   %.4f ms\n", t);

    t = bench([&]{
        cudaLaunchKernelExC(&cfg_pdl, (void*)p_signal_early, out_args);
        cudaLaunchKernelExC(&cfg_plain, (void*)c_with_wait_no_work, out_args);
    });
    printf("PDL (p_signal_early + WAIT-only):        %.4f ms\n", t);

    CK(cudaFree(d_out));
    return 0;
}
