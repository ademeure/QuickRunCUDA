// PDL scenarios: where does PDL actually HELP?
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef ITERS_P
#define ITERS_P 200000
#endif

extern "C" __global__ void p_compute(float *out, int signal_at) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = signal_at;
    int half2 = ITERS_P - signal_at;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

// Tiny compute consumer
extern "C" __global__ void c_tiny(float *out, int iters) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0002f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void c_tiny_nopdl(float *out, int iters) {
    float a = 2.0f + (threadIdx.x & 31) * 0.002f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0002f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

// Memory-bound consumer (LDG-heavy)
extern "C" __global__ void c_memory(float *in, float *out, int n_loads) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int i = 0; i < n_loads; i++) {
        int idx = (tid + i * stride) & ((64 << 20) - 1);  // wrap to 256MB
        acc += in[idx];
    }
    if (acc == -42.0f) out[tid] = acc;
}

extern "C" __global__ void c_memory_nopdl(float *in, float *out, int n_loads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int i = 0; i < n_loads; i++) {
        int idx = (tid + i * stride) & ((64 << 20) - 1);
        acc += in[idx];
    }
    if (acc == -42.0f) out[tid] = acc;
}

// Producer with non-uniform tail (some blocks much shorter)
// Half blocks do half work, half do full work
extern "C" __global__ void p_uneven(float *out, int signal_at) {
    int my_iters = (blockIdx.x % 2 == 0) ? (ITERS_P / 2) : ITERS_P;
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    int half1 = (signal_at < my_iters) ? signal_at : my_iters;
    int half2 = my_iters - half1;
    #pragma unroll 1
    for (int i = 0; i < half1; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = 0; i < half2; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void p_uneven_nopdl(float *out) {
    int my_iters = (blockIdx.x % 2 == 0) ? (ITERS_P / 2) : ITERS_P;
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < my_iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (a == -42.0f) out[blockIdx.x*blockDim.x + threadIdx.x] = a;
}

int main(int argc, char **argv) {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;
    int blocks = sm_count, threads = 128;

    printf("# B300 PDL scenarios: %d blocks x %d threads, ITERS_P=%d\n",
           blocks, threads, ITERS_P);

    float *d_in, *d_out;
    size_t buf = 256 << 20;  // 256MB
    CK(cudaMalloc(&d_in, buf));
    CK(cudaMalloc(&d_out, buf));
    CK(cudaMemset(d_in, 0x40, buf));
    CK(cudaMemset(d_out, 0, buf));

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

    int sig_full = ITERS_P;  // = signal at end (no PDL benefit)
    int sig_zero = 0;        // = signal immediately

    // ===== Scenario 1: Tiny consumer (should hide in producer tail) =====
    printf("\n## Scenario 1: Compute producer + tiny compute consumer (sweep consumer size)\n");
    int c_sizes[] = {100, 1000, 5000, 10000, 25000, 50000, 100000};
    for (int ci = 0; ci < 7; ci++) {
        int csz = c_sizes[ci];
        // Sequential
        float t_seq = bench([&]{
            p_compute<<<blocks,threads,0,s>>>(d_out, sig_full);  // no real PDL
            c_tiny_nopdl<<<blocks,threads,0,s>>>(d_out, csz);
        });
        // Producer alone
        float t_p = bench([&]{
            p_compute<<<blocks,threads,0,s>>>(d_out, sig_full);
        });
        // PDL with signal at 0%
        float t_pdl0 = bench([&]{
            void *p_args[] = {&d_out, &sig_zero};
            void *c_args[] = {&d_out, &csz};
            cudaLaunchKernelExC(&cfg_pdl, (void*)p_compute, p_args);
            cudaLaunchKernelExC(&cfg_plain, (void*)c_tiny, c_args);
        });
        printf("  c_iters=%-7d : p_alone=%.3f  seq=%.3f  pdl0=%.3f  saved=%+.3f ms (%+.1f%%)\n",
               csz, t_p, t_seq, t_pdl0, t_seq - t_pdl0, 100.f*(t_seq-t_pdl0)/t_seq);
    }

    // ===== Scenario 2: Memory-bound consumer =====
    printf("\n## Scenario 2: Compute producer + MEMORY consumer\n");
    int n_loads_arr[] = {1000, 5000, 10000, 50000};
    for (int li = 0; li < 4; li++) {
        int nl = n_loads_arr[li];
        float t_seq = bench([&]{
            p_compute<<<blocks,threads,0,s>>>(d_out, sig_full);
            c_memory_nopdl<<<blocks,threads,0,s>>>(d_in, d_out, nl);
        });
        float t_pdl = bench([&]{
            void *p_args[] = {&d_out, &sig_zero};
            void *c_args[] = {&d_in, &d_out, &nl};
            cudaLaunchKernelExC(&cfg_pdl, (void*)p_compute, p_args);
            cudaLaunchKernelExC(&cfg_plain, (void*)c_memory, c_args);
        });
        printf("  n_loads=%-6d : seq=%.3f  pdl0=%.3f  saved=%+.3f ms (%+.1f%%)\n",
               nl, t_seq, t_pdl, t_seq - t_pdl, 100.f*(t_seq-t_pdl)/t_seq);
    }

    // ===== Scenario 3: Producer with stragglers =====
    printf("\n## Scenario 3: Uneven producer (half blocks 50%% work) + tiny consumer\n");
    for (int ci = 0; ci < 7; ci++) {
        int csz = c_sizes[ci];
        float t_seq = bench([&]{
            p_uneven_nopdl<<<blocks,threads,0,s>>>(d_out);
            c_tiny_nopdl<<<blocks,threads,0,s>>>(d_out, csz);
        });
        float t_p = bench([&]{
            p_uneven_nopdl<<<blocks,threads,0,s>>>(d_out);
        });
        // PDL: signal in even blocks (after their half-work) - slower blocks signal at midpoint
        int sig_mid = ITERS_P / 4;  // even blocks signal at midpoint of their work
        float t_pdl = bench([&]{
            void *p_args[] = {&d_out, &sig_mid};
            void *c_args[] = {&d_out, &csz};
            cudaLaunchKernelExC(&cfg_pdl, (void*)p_uneven, p_args);
            cudaLaunchKernelExC(&cfg_plain, (void*)c_tiny, c_args);
        });
        printf("  c_iters=%-7d : p_alone=%.3f  seq=%.3f  pdl=%.3f  saved=%+.3f ms (%+.1f%%)\n",
               csz, t_p, t_seq, t_pdl, t_seq - t_pdl, 100.f*(t_seq-t_pdl)/t_seq);
    }

    CK(cudaFree(d_in));
    CK(cudaFree(d_out));
    return 0;
}
