// PDL with ProgrammaticEvent: more flexible than ProgrammaticStreamSerialization
// allows event-based dependent launches across streams
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef ITERS_K
#define ITERS_K 5000
#endif

extern "C" __global__ void k_pdl(float *out, int signal_at, int sentinel) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + sentinel * 0.00001f;
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
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void k_pdl_first(float *out, int signal_at, int sentinel) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + sentinel * 0.00001f;
    int half1 = signal_at;
    int half2 = ITERS_K - signal_at;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void k_nopdl(float *out, int sentinel) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + sentinel * 0.00001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_K; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;
    int blocks = sm_count, threads = 128;

    printf("# B300 PDL with ProgrammaticEvent attribute\n");
    printf("# %d blocks x %d threads, %d iters/kernel\n", blocks, threads, ITERS_K);

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));
    CK(cudaMemset(d_out, 0, blocks * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    cudaLaunchAttribute attr_pdl;
    attr_pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr_pdl.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_pdl = {dim3(blocks), dim3(threads), 0, s, &attr_pdl, 1};

    auto bench_sync = [&](auto fn, int trials=10) {
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

    int chain = 32;
    int sig_99 = (ITERS_K * 99) / 100;
    int sig_100 = ITERS_K;

    printf("\n## Test 1: PDL chain in single stream (sweep signal)\n");
    {
        // No PDL
        int sentinel = 0;
        float t_nopdl = bench_sync([&]{
            for (int k = 0; k < chain; k++)
                k_nopdl<<<blocks,threads,0,s>>>(d_out, k);
        });
        printf("  no_pdl, %d kernels: %.3f ms (%.2f us/kernel)\n",
               chain, t_nopdl, t_nopdl*1000/chain);

        for (int pct : {50, 75, 90, 95, 98, 99, 100}) {
            int sig = (ITERS_K * pct) / 100;
            float t = bench_sync([&]{
                void *first_args[] = {&d_out, &sig, &sentinel};
                cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl_first, first_args);
                for (int k = 1; k < chain; k++) {
                    int sk = k;
                    void *kargs[] = {&d_out, &sig, &sk};
                    cudaLaunchKernelExC(&cfg_pdl, (void*)k_pdl, kargs);
                }
            });
            printf("  pdl_sig_%-3d, %d kernels: %.3f ms (%.2f us/kernel, save=%+.2f us/kernel)\n",
                   pct, chain, t, t*1000/chain, (t_nopdl-t)*1000/chain);
        }
    }

    // ===== Test 2: ProgrammaticEvent (cross-stream PDL) =====
    printf("\n## Test 2: ProgrammaticEvent — cross-stream dependent launch\n");
    {
        cudaStream_t s_prod, s_cons;
        CK(cudaStreamCreateWithFlags(&s_prod, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&s_cons, cudaStreamNonBlocking));

        cudaEvent_t pdl_event;
        CK(cudaEventCreateWithFlags(&pdl_event, cudaEventDisableTiming));

        // Producer: signal a programmatic event
        cudaLaunchAttribute prod_attrs[1];
        prod_attrs[0].id = cudaLaunchAttributeProgrammaticEvent;
        prod_attrs[0].val.programmaticEvent.event = pdl_event;
        prod_attrs[0].val.programmaticEvent.flags = 0;
        prod_attrs[0].val.programmaticEvent.triggerAtBlockStart = 0;

        cudaLaunchConfig_t cfg_prod = {dim3(blocks), dim3(threads), 0, s_prod, prod_attrs, 1};

        // Consumer launches when programmatic event signals
        cudaLaunchConfig_t cfg_cons = {dim3(blocks), dim3(threads), 0, s_cons, nullptr, 0};

        // Sequential baseline (cross-stream with explicit event)
        cudaEvent_t e_sync; CK(cudaEventCreate(&e_sync));
        int sentinel0 = 0, sentinel1 = 1;
        float t_seq_2stream = bench_sync([&]{
            k_nopdl<<<blocks,threads,0,s_prod>>>(d_out, sentinel0);
            cudaEventRecord(e_sync, s_prod);
            cudaStreamWaitEvent(s_cons, e_sync, 0);
            k_nopdl<<<blocks,threads,0,s_cons>>>(d_out, sentinel1);
        });

        // PDL with ProgrammaticEvent
        int sig = sig_99;
        // For ProgrammaticEvent: stream waits on event from another stream
        // The event is "fired" by producer's griddepcontrol.launch_dependents
        float t_pdl_event = bench_sync([&]{
            void *p_args[] = {&d_out, &sig, &sentinel0};
            cudaLaunchKernelExC(&cfg_prod, (void*)k_pdl_first, p_args);
            cudaStreamWaitEvent(s_cons, pdl_event, cudaEventWaitDefault);
            void *c_args[] = {&d_out, &sig, &sentinel1};
            cudaLaunchKernelExC(&cfg_cons, (void*)k_pdl, c_args);
        });

        printf("  seq 2-stream w/event sync: %.4f ms\n", t_seq_2stream);
        printf("  PDL ProgrammaticEvent:     %.4f ms (saves %+.4f ms)\n",
               t_pdl_event, t_seq_2stream - t_pdl_event);

        cudaStreamDestroy(s_prod);
        cudaStreamDestroy(s_cons);
        cudaEventDestroy(pdl_event);
        cudaEventDestroy(e_sync);
    }

    CK(cudaFree(d_out));
    return 0;
}
