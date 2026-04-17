// Nested PDL: A → B → C with PDL at each stage
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define ITERS 5000

extern "C" __global__ void k_first(float *out, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at, half2 = ITERS - signal_at;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void k_middle(float *out, int signal_at) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    int half1 = signal_at, half2 = ITERS - signal_at;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void k_last(float *out) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float a = 3.0f + (threadIdx.x & 31) * 0.003f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void k_nopdl(float *out) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    printf("# B300 nested PDL test (3-kernel chain A→B→C)\n");
    printf("# %d blocks × %d threads, ITERS=%d\n\n", blocks, threads, ITERS);

    float *d_out;
    CK(cudaMalloc(&d_out, blocks * sizeof(float)));

    cudaStream_t s; CK(cudaStreamCreate(&s));

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {dim3(blocks), dim3(threads), 0, s, &attr, 1};

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

    int sig = (ITERS * 99) / 100;

    // Sequential 3-kernel chain
    float t_seq = bench([&]{
        k_nopdl<<<blocks, threads, 0, s>>>(d_out);
        k_nopdl<<<blocks, threads, 0, s>>>(d_out);
        k_nopdl<<<blocks, threads, 0, s>>>(d_out);
    });

    // Full PDL chain: A signals B, B signals C
    float t_pdl_full = bench([&]{
        void *fa[] = {&d_out, &sig};
        cudaLaunchKernelExC(&cfg, (void*)k_first, fa);
        void *ma[] = {&d_out, &sig};
        cudaLaunchKernelExC(&cfg, (void*)k_middle, ma);
        { void *la[] = {&d_out}; cudaLaunchKernelExC(&cfg, (void*)k_last, la); };
    });

    // Only A signals B (B no signal to C)
    float t_pdl_AB = bench([&]{
        void *fa[] = {&d_out, &sig};
        cudaLaunchKernelExC(&cfg, (void*)k_first, fa);
        void *ma[] = {&d_out, &sig};
        cudaLaunchKernelExC(&cfg, (void*)k_middle, ma);
        // C without PDL launch attr — sequential after B
        k_nopdl<<<blocks, threads, 0, s>>>(d_out);
    });

    // Only B signals C (A doesn't signal)
    float t_pdl_BC = bench([&]{
        k_nopdl<<<blocks, threads, 0, s>>>(d_out);
        void *ma[] = {&d_out, &sig};
        cudaLaunchKernelExC(&cfg, (void*)k_middle, ma);
        { void *la[] = {&d_out}; cudaLaunchKernelExC(&cfg, (void*)k_last, la); };
    });

    printf("Sequential 3-chain (no PDL):           %.4f ms\n", t_seq);
    printf("Full PDL (A→B→C, all signal):          %.4f ms (save %+.4f)\n", t_pdl_full, t_seq - t_pdl_full);
    printf("PDL A→B only (C plain):                %.4f ms (save %+.4f)\n", t_pdl_AB, t_seq - t_pdl_AB);
    printf("PDL B→C only (A plain):                %.4f ms (save %+.4f)\n", t_pdl_BC, t_seq - t_pdl_BC);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
