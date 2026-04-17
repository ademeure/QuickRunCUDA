// Critical PDL semantic test: does ONE block signal fire PDL, or do ALL blocks need to?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define ITERS 50000

// Producer: ALL blocks signal at signal_iter
extern "C" __global__ void producer_all_signal(float *out, int signal_iter) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_iter, half2 = ITERS - signal_iter;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // ALL blocks signal
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

// Producer: ONLY ONE block signals (block 0)
extern "C" __global__ void producer_one_signal(float *out, int signal_iter) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_iter, half2 = ITERS - signal_iter;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (blockIdx.x == 0) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // only block 0 signals
    }
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

// Producer: NO blocks signal
extern "C" __global__ void producer_no_signal(float *out, int signal_iter) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_iter, half2 = ITERS - signal_iter;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    // No griddepcontrol.launch_dependents
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void consumer(float *out) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    for (int i = 0; i < ITERS; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    printf("# B300 PDL signal-count semantic test\n");
    printf("# %d blocks × %d threads, ITERS=%d\n\n", blocks, threads, ITERS);

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));

    cudaLaunchAttribute attr_pdl;
    attr_pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr_pdl.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_pdl = {dim3(blocks), dim3(threads), 0, s, &attr_pdl, 1};

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

    int sig = (ITERS * 50) / 100;  // signal at midpoint

    // Sequential baseline (no PDL launch attr)
    float t_seq = bench([&]{
        producer_no_signal<<<blocks, threads, 0, s>>>(d_out, sig);
        consumer<<<blocks, threads, 0, s>>>(d_out);
    });

    // ALL blocks signal
    float t_all = bench([&]{
        void *pa[] = {&d_out, &sig};
        cudaLaunchKernelExC(&cfg_pdl, (void*)producer_all_signal, pa);
        consumer<<<blocks, threads, 0, s>>>(d_out);
    });

    // ONE block signals
    float t_one = bench([&]{
        void *pa[] = {&d_out, &sig};
        cudaLaunchKernelExC(&cfg_pdl, (void*)producer_one_signal, pa);
        consumer<<<blocks, threads, 0, s>>>(d_out);
    });

    // NO block signals (PDL launch attr but never signaled)
    float t_none = bench([&]{
        void *pa[] = {&d_out, &sig};
        cudaLaunchKernelExC(&cfg_pdl, (void*)producer_no_signal, pa);
        consumer<<<blocks, threads, 0, s>>>(d_out);
    });

    printf("# producer + consumer pair, signal at 50%% of ITERS=%d:\n", ITERS);
    printf("  sequential (no PDL):           %.4f ms\n", t_seq);
    printf("  PDL: ALL blocks signal:        %.4f ms (diff %+.4f)\n",
           t_all, t_all - t_seq);
    printf("  PDL: ONE block signals (blk 0): %.4f ms (diff %+.4f)\n",
           t_one, t_one - t_seq);
    printf("  PDL: NO blocks signal:         %.4f ms (diff %+.4f)\n",
           t_none, t_none - t_seq);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
