// PDL (Programmatic Dependent Launch) benchmark for B300
// Tests overlap savings via griddepcontrol.launch_dependents / .wait
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

// Producer: split into two clean FFMA loops with signal in between (no per-iter if)
extern "C" __global__ void producer(float *out, int signal_iter) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    float b = 1.0001f, c = 0.0001f;

    int half1 = signal_iter;
    int half2 = ITERS_P - signal_iter;

    #pragma unroll 1
    for (int i = 0; i < half1; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(b), "f"(c));
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(b), "f"(c));
    }
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

// Producer that signals at the very end (signal_pct=100) — no overlap
extern "C" __global__ void producer_late_signal(float *out) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    float b = 1.0001f, c = 0.0001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_P; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(b), "f"(c));
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

// Producer that signals immediately (signal_pct=0) — full overlap potential
extern "C" __global__ void producer_early_signal(float *out) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    float b = 1.0001f, c = 0.0001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_P; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(b), "f"(c));
    }
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

// Consumer: starts with wait, then ITERS_C FFMAs
extern "C" __global__ void consumer(float *out) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    float b = 1.0001f, c = 0.0002f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_C; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(b), "f"(c));
    }
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

// Producer WITHOUT PDL signal (sequential baseline)
extern "C" __global__ void producer_nopdl(float *out) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    float b = 1.0001f, c = 0.0001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_P; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(b), "f"(c));
    }
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

// Consumer WITHOUT wait (sequential baseline)
extern "C" __global__ void consumer_nopdl(float *out) {
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    float b = 1.0001f, c = 0.0002f;
    #pragma unroll 1
    for (int i = 0; i < ITERS_C; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(b), "f"(c));
    }
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

int main(int argc, char **argv) {
    int dev = 0;
    CK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    int sm_count = prop.multiProcessorCount;

    int blocks = sm_count;
    int threads = 128;
    if (argc > 1) blocks = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    printf("# B300 PDL bench: %d SMs, %d blocks x %d threads, ITERS_P=%d, ITERS_C=%d\n",
           sm_count, blocks, threads, ITERS_P, ITERS_C);

    float *d_out;
    size_t out_sz = blocks * threads * sizeof(float);
    CK(cudaMalloc(&d_out, out_sz));
    CK(cudaMemset(d_out, 0, out_sz));

    cudaStream_t s;
    CK(cudaStreamCreate(&s));

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    const int N_TRIALS = 10;

    auto bench = [&](auto launch_fn) {
        float best = 1e30f;
        for (int t = 0; t < N_TRIALS; t++) {
            CK(cudaEventRecord(e0, s));
            launch_fn();
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms < best) best = ms;
        }
        return best;
    };

    // Warmup
    for (int i = 0; i < 3; i++) {
        producer_nopdl<<<blocks, threads, 0, s>>>(d_out);
        consumer_nopdl<<<blocks, threads, 0, s>>>(d_out);
    }
    CK(cudaStreamSynchronize(s));

    // ===== Baselines =====
    float t_p_alone = bench([&]() {
        producer_nopdl<<<blocks, threads, 0, s>>>(d_out);
    });
    float t_c_alone = bench([&]() {
        consumer_nopdl<<<blocks, threads, 0, s>>>(d_out);
    });
    float t_seq = bench([&]() {
        producer_nopdl<<<blocks, threads, 0, s>>>(d_out);
        consumer_nopdl<<<blocks, threads, 0, s>>>(d_out);
    });

    printf("\n# Baselines (no PDL):\n");
    printf("producer_alone_ms  = %.4f\n", t_p_alone);
    printf("consumer_alone_ms  = %.4f\n", t_c_alone);
    printf("sequential_ms      = %.4f (P+C)\n", t_seq);
    printf("p_plus_c_sum_ms    = %.4f\n", t_p_alone + t_c_alone);
    printf("inter_kernel_gap   = %+.4f ms\n", t_seq - (t_p_alone + t_c_alone));

    // ===== PDL launch attributes =====
    cudaLaunchAttribute attr_pdl;
    attr_pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr_pdl.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t cfg_pdl = {};
    cfg_pdl.gridDim = dim3(blocks);
    cfg_pdl.blockDim = dim3(threads);
    cfg_pdl.stream = s;
    cfg_pdl.attrs = &attr_pdl;
    cfg_pdl.numAttrs = 1;

    cudaLaunchConfig_t cfg_plain = {};
    cfg_plain.gridDim = dim3(blocks);
    cfg_plain.blockDim = dim3(threads);
    cfg_plain.stream = s;

    // ===== Test: producer alone WITH PDL launch attribute (overhead?) =====
    {
        // Warmup
        for (int i = 0; i < 3; i++) {
            void *p_args[] = {&d_out};
            CK(cudaLaunchKernelExC(&cfg_pdl, (void*)producer_late_signal, p_args));
        }
        CK(cudaStreamSynchronize(s));

        float t_p_pdl_alone = bench([&]() {
            void *p_args[] = {&d_out};
            cudaLaunchKernelExC(&cfg_pdl, (void*)producer_late_signal, p_args);
        });
        printf("\n# PDL launch attribute overhead (no consumer):\n");
        printf("producer_pdl_alone_ms = %.4f (vs %.4f no-PDL)\n", t_p_pdl_alone, t_p_alone);
    }

    // ===== PDL Signal Sweep =====
    printf("\n# PDL Signal Point Sweep (P=%d iters, C=%d iters):\n", ITERS_P, ITERS_C);
    printf("# %-8s %-12s %-12s %-12s %-12s\n",
           "pct", "time_ms", "vs_seq", "savings_ms", "savings_pct");

    int signal_pcts[] = {0, 5, 10, 25, 50, 75, 90, 95, 100};
    int n_pcts = sizeof(signal_pcts) / sizeof(int);

    for (int pi = 0; pi < n_pcts; pi++) {
        int pct = signal_pcts[pi];
        int signal_iter = (ITERS_P * pct) / 100;

        // Warmup
        for (int i = 0; i < 3; i++) {
            void *p_args[] = {&d_out, &signal_iter};
            void *c_args[] = {&d_out};
            CK(cudaLaunchKernelExC(&cfg_pdl, (void*)producer, p_args));
            CK(cudaLaunchKernelExC(&cfg_plain, (void*)consumer, c_args));
        }
        CK(cudaStreamSynchronize(s));

        float t = bench([&]() {
            void *p_args[] = {&d_out, &signal_iter};
            void *c_args[] = {&d_out};
            cudaLaunchKernelExC(&cfg_pdl, (void*)producer, p_args);
            cudaLaunchKernelExC(&cfg_plain, (void*)consumer, c_args);
        });

        float savings = t_seq - t;
        printf("  %-8d %-12.4f %-12.4f %+-12.4f %+-12.2f%%\n",
               pct, t, t / t_seq, savings, 100.0f * savings / t_seq);
    }

    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    CK(cudaStreamDestroy(s));
    CK(cudaFree(d_out));
    return 0;
}
