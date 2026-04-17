// Realistic LLM-serving-like pipeline: GEMM-compute → softmax → GEMM-compute
// Tests PDL benefit on a more realistic chain
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define COMPUTE_ITERS 10000  // ~120 us per kernel
#define MEM_LOADS 1000       // memory-bound

// "GEMM" - compute-bound
extern "C" __global__ void gemm_pdl(float *out, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at, half2 = COMPUTE_ITERS - signal_at;
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

extern "C" __global__ void gemm_pdl_first(float *out, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at, half2 = COMPUTE_ITERS - signal_at;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void gemm_nopdl(float *out) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < COMPUTE_ITERS; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

// "Softmax" - memory-bound LDG/STG
extern "C" __global__ void softmax_pdl(float *src, float *dst, int N, int signal_at) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc = 0;
    for (int i = tid; i < signal_at; i += stride) {
        acc += src[i];
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    for (int i = signal_at + tid; i < N; i += stride) {
        acc += src[i];
    }
    if (tid < N) dst[tid] = acc / N;
}

extern "C" __global__ void softmax_nopdl(float *src, float *dst, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0;
    for (int i = tid; i < N; i += stride) {
        acc += src[i];
    }
    if (tid < N) dst[tid] = acc / N;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount, threads = 128;

    printf("# B300 LLM-style pipeline: GEMM → softmax → GEMM\n");
    printf("# COMPUTE_ITERS=%d, MEM_LOADS=%d\n", COMPUTE_ITERS, MEM_LOADS);

    int N = 4 << 20;
    float *d_buf1, *d_buf2;
    CK(cudaMalloc(&d_buf1, N * sizeof(float)));
    CK(cudaMalloc(&d_buf2, N * sizeof(float)));
    CK(cudaMemset(d_buf1, 0x40, N * sizeof(float)));

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

    int n_layers = 16;  // 16 transformer layers, each = GEMM + softmax + GEMM = 3 kernels
    int total_kernels = n_layers * 3;

    int sig_compute = (COMPUTE_ITERS * 99) / 100;
    int sig_mem = N * 99 / 100;

    // ===== Sequential (no PDL) =====
    float t_seq = bench([&]{
        for (int l = 0; l < n_layers; l++) {
            gemm_nopdl<<<blocks, threads, 0, s>>>(d_buf2);
            softmax_nopdl<<<blocks, threads, 0, s>>>(d_buf1, d_buf2, N);
            gemm_nopdl<<<blocks, threads, 0, s>>>(d_buf1);
        }
    });

    // ===== With PDL =====
    float t_pdl = bench([&]{
        for (int l = 0; l < n_layers; l++) {
            void *ga[] = {&d_buf2, &sig_compute};
            cudaLaunchKernelExC(&cfg_pdl, (void*)gemm_pdl_first, ga);
            void *sa[] = {&d_buf1, &d_buf2, &N, &sig_mem};
            cudaLaunchKernelExC(&cfg_pdl, (void*)softmax_pdl, sa);
            void *ga2[] = {&d_buf1, &sig_compute};
            cudaLaunchKernelExC(&cfg_pdl, (void*)gemm_pdl, ga2);
        }
    });

    // ===== With CUDA Graph (no PDL) =====
    cudaGraph_t g; cudaGraphExec_t g_exec;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < n_layers; l++) {
        gemm_nopdl<<<blocks, threads, 0, s>>>(d_buf2);
        softmax_nopdl<<<blocks, threads, 0, s>>>(d_buf1, d_buf2, N);
        gemm_nopdl<<<blocks, threads, 0, s>>>(d_buf1);
    }
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&g_exec, g, nullptr, nullptr, 0);
    float t_graph = bench([&]{ cudaGraphLaunch(g_exec, s); });

    // ===== With CUDA Graph + PDL =====
    cudaGraph_t g2; cudaGraphExec_t g2_exec;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < n_layers; l++) {
        void *ga[] = {&d_buf2, &sig_compute};
        cudaLaunchKernelExC(&cfg_pdl, (void*)gemm_pdl_first, ga);
        void *sa[] = {&d_buf1, &d_buf2, &N, &sig_mem};
        cudaLaunchKernelExC(&cfg_pdl, (void*)softmax_pdl, sa);
        void *ga2[] = {&d_buf1, &sig_compute};
        cudaLaunchKernelExC(&cfg_pdl, (void*)gemm_pdl, ga2);
    }
    cudaStreamEndCapture(s, &g2);
    cudaGraphInstantiate(&g2_exec, g2, nullptr, nullptr, 0);
    float t_graph_pdl = bench([&]{ cudaGraphLaunch(g2_exec, s); });

    printf("\n# %d layers × 3 kernels = %d total kernels\n", n_layers, total_kernels);
    printf("  sequential (no PDL):   %.3f ms (%.2f us/kernel)\n",
           t_seq, t_seq*1000/total_kernels);
    printf("  PDL only:              %.3f ms (%.2f us/kernel) save=%+.2f us/kernel\n",
           t_pdl, t_pdl*1000/total_kernels, (t_seq-t_pdl)*1000/total_kernels);
    printf("  CUDA Graph only:       %.3f ms (%.2f us/kernel) save=%+.2f us/kernel\n",
           t_graph, t_graph*1000/total_kernels, (t_seq-t_graph)*1000/total_kernels);
    printf("  Graph + PDL:           %.3f ms (%.2f us/kernel) save=%+.2f us/kernel\n",
           t_graph_pdl, t_graph_pdl*1000/total_kernels, (t_seq-t_graph_pdl)*1000/total_kernels);

    cudaGraphExecDestroy(g_exec); cudaGraphDestroy(g);
    cudaGraphExecDestroy(g2_exec); cudaGraphDestroy(g2);
    cudaStreamDestroy(s);
    cudaFree(d_buf1); cudaFree(d_buf2);
    return 0;
}
