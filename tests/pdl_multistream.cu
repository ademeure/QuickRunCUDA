// PDL multi-stream test: ProgrammaticEvent for cross-stream dependent launch
// Key question: can PDL signal go to a different stream?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef ITERS_K
#define ITERS_K 5000
#endif

extern "C" __global__ void producer(float *out, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at, half2 = ITERS_K - signal_at;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void consumer(float *out) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_K; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0002f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void producer_nopdl(float *out) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_K; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void consumer_nopdl(float *out) {
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_K; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0002f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    printf("# B300 PDL multi-stream test\n");
    printf("# %d blocks × %d threads, %d iters each\n\n", blocks, threads, ITERS_K);

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * threads * sizeof(float)));
    CK(cudaMemset(d_out, 0, blocks * threads * sizeof(float)));

    cudaStream_t s_prod, s_cons;
    CK(cudaStreamCreateWithFlags(&s_prod, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s_cons, cudaStreamNonBlocking));
    cudaStream_t s_single;
    CK(cudaStreamCreateWithFlags(&s_single, cudaStreamNonBlocking));

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

    int sig_99 = (ITERS_K * 99) / 100;
    int sig_50 = ITERS_K / 2;

    // ===== Test 1: Single stream sequential (no PDL) =====
    float t_seq_single = bench([&]{
        producer_nopdl<<<blocks,threads,0,s_single>>>(d_out);
        consumer_nopdl<<<blocks,threads,0,s_single>>>(d_out);
    });
    printf("## Single stream sequential (no PDL): %.4f ms\n", t_seq_single);

    // ===== Test 2: Single stream PDL =====
    {
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {dim3(blocks), dim3(threads), 0, s_single, &attr, 1};

        float t_pdl_single_99 = bench([&]{
            void *pa[] = {&d_out, &sig_99};
            cudaLaunchKernelExC(&cfg, (void*)producer, pa);
            void *ca[] = {&d_out};
            consumer<<<blocks,threads,0,s_single>>>(d_out);
        });
        float t_pdl_single_50 = bench([&]{
            void *pa[] = {&d_out, &sig_50};
            cudaLaunchKernelExC(&cfg, (void*)producer, pa);
            consumer<<<blocks,threads,0,s_single>>>(d_out);
        });

        printf("## Single stream PDL (sig 99%%): %.4f ms (save %+.4f ms)\n",
               t_pdl_single_99, t_seq_single - t_pdl_single_99);
        printf("## Single stream PDL (sig 50%%): %.4f ms (save %+.4f ms)\n",
               t_pdl_single_50, t_seq_single - t_pdl_single_50);
    }

    // ===== Test 3: Two streams with cudaStreamWaitEvent (no PDL) =====
    {
        cudaEvent_t evt; CK(cudaEventCreate(&evt));
        float t_two_event = bench([&]{
            producer_nopdl<<<blocks,threads,0,s_prod>>>(d_out);
            cudaEventRecord(evt, s_prod);
            cudaStreamWaitEvent(s_cons, evt, 0);
            consumer_nopdl<<<blocks,threads,0,s_cons>>>(d_out);
        });
        printf("\n## Two streams, event-based sync: %.4f ms (vs 1-stream %.4f, diff=%+.4f)\n",
               t_two_event, t_seq_single, t_two_event - t_seq_single);
        cudaEventDestroy(evt);
    }

    // ===== Test 4: Two streams with PDL ProgrammaticEvent =====
    {
        cudaEvent_t pdl_event;
        CK(cudaEventCreateWithFlags(&pdl_event, cudaEventDisableTiming));

        cudaLaunchAttribute prod_attrs[1];
        prod_attrs[0].id = cudaLaunchAttributeProgrammaticEvent;
        prod_attrs[0].val.programmaticEvent.event = pdl_event;
        prod_attrs[0].val.programmaticEvent.flags = 0;
        prod_attrs[0].val.programmaticEvent.triggerAtBlockStart = 0;
        cudaLaunchConfig_t cfg_prod = {dim3(blocks), dim3(threads), 0, s_prod, prod_attrs, 1};

        float t_pdl_2stream_99 = bench([&]{
            void *pa[] = {&d_out, &sig_99};
            cudaLaunchKernelExC(&cfg_prod, (void*)producer, pa);
            cudaStreamWaitEvent(s_cons, pdl_event, cudaEventWaitDefault);
            consumer<<<blocks,threads,0,s_cons>>>(d_out);
        });
        float t_pdl_2stream_50 = bench([&]{
            void *pa[] = {&d_out, &sig_50};
            cudaLaunchKernelExC(&cfg_prod, (void*)producer, pa);
            cudaStreamWaitEvent(s_cons, pdl_event, cudaEventWaitDefault);
            consumer<<<blocks,threads,0,s_cons>>>(d_out);
        });

        printf("\n## Two streams PDL ProgrammaticEvent (sig 99%%): %.4f ms (save %+.4f vs 1-stream PDL)\n",
               t_pdl_2stream_99, t_seq_single - t_pdl_2stream_99);
        printf("## Two streams PDL ProgrammaticEvent (sig 50%%): %.4f ms (save %+.4f)\n",
               t_pdl_2stream_50, t_seq_single - t_pdl_2stream_50);

        cudaEventDestroy(pdl_event);
    }

    // ===== Test 5: Triggered at block start =====
    {
        cudaEvent_t pdl_event;
        CK(cudaEventCreateWithFlags(&pdl_event, cudaEventDisableTiming));

        cudaLaunchAttribute prod_attrs[1];
        prod_attrs[0].id = cudaLaunchAttributeProgrammaticEvent;
        prod_attrs[0].val.programmaticEvent.event = pdl_event;
        prod_attrs[0].val.programmaticEvent.flags = 0;
        prod_attrs[0].val.programmaticEvent.triggerAtBlockStart = 1;  // ← trigger when blocks start
        cudaLaunchConfig_t cfg_prod = {dim3(blocks), dim3(threads), 0, s_prod, prod_attrs, 1};

        float t_block_start = bench([&]{
            // sig irrelevant since event triggers when blocks START
            void *pa[] = {&d_out, &sig_99};
            cudaLaunchKernelExC(&cfg_prod, (void*)producer, pa);
            cudaStreamWaitEvent(s_cons, pdl_event, cudaEventWaitDefault);
            consumer<<<blocks,threads,0,s_cons>>>(d_out);
        });

        printf("\n## Two streams PDL triggerAtBlockStart=1: %.4f ms (save %+.4f)\n",
               t_block_start, t_seq_single - t_block_start);

        cudaEventDestroy(pdl_event);
    }

    // ===== Test 6: PDL + parallel streams (no dependency) =====
    {
        // Just two independent streams, no PDL
        float t_par = bench([&]{
            producer_nopdl<<<blocks,threads,0,s_prod>>>(d_out);
            consumer_nopdl<<<blocks,threads,0,s_cons>>>(d_out);
        });
        printf("\n## Two streams parallel (NO sync, may not be valid for chains!): %.4f ms\n", t_par);
    }

    cudaStreamDestroy(s_prod);
    cudaStreamDestroy(s_cons);
    cudaStreamDestroy(s_single);
    CK(cudaFree(d_out));
    return 0;
}
