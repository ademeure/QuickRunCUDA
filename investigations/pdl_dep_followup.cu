// PDL dep-chain follow-up: understand the dep chain anomaly
// Scenario 3 showed ~0 us/kernel benefit at n=8-32 but +1.47 us at n=128
// This isolates whether the issue is griddepcontrol.wait overhead
// or simply that short dep-chains don't benefit (launch overhead already hidden by prev)
//
// Also tests: does wait BEFORE first read add fence-like overhead?
// Compare: wait, then read vs. wait-less, then read

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d [%s]: %s\n",__FILE__,__LINE__,#c,cudaGetErrorString(e)); exit(1);} } while(0)

// ---- Kernel that reads, computes, writes (no PDL) ----
__global__ void k_dep_plain(const float *in, float *out, int iters, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float a = in[tid];
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    out[tid] = a;
}

// ---- PDL variants ----
__global__ void k_dep_pdl_first(const float *in, float *out, int iters, int N, int sig) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float a = in[tid];
    #pragma unroll 1
    for (int i = 0; i < sig; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = sig; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    out[tid] = a;
}

__global__ void k_dep_pdl_mid(const float *in, float *out, int iters, int N, int sig) {
    // Uses both wait AND launch_dependents (mid-chain kernel)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float a = in[tid];
    #pragma unroll 1
    for (int i = 0; i < sig; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    #pragma unroll 1
    for (int i = sig; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    out[tid] = a;
}

// ---- Write-tile + wait variants to test wait overhead independently ----
// Full compute, full write, WITH griddepcontrol.wait but NO launch_dependents
// (i.e., consumer-only behavior — waits but doesn't forward)
__global__ void k_wait_only(float *out, int iters) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    for (int j = 0; j < 4; j++) out[tid * 4 + j] = a + j * 0.01f;
}

__global__ void k_signal_only(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    for (int j = 0; j < 4; j++) out[tid * 4 + j] = a + j * 0.01f;  // write first
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (a == -42.f) out[0] = a;  // keep acc live
}

__global__ void k_nowait(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0f + (threadIdx.x & 31) * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    for (int j = 0; j < 4; j++) out[tid * 4 + j] = a + j * 0.01f;
}

template<typename Fn>
float bench_fn(Fn fn, cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1, int warmup=3, int trials=15) {
    for (int i = 0; i < warmup; i++) fn();
    CK(cudaStreamSynchronize(s));
    float best = 1e30f;
    for (int t = 0; t < trials; t++) {
        CK(cudaEventRecord(e0, s));
        fn();
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best) best = ms;
    }
    return best;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;
    int blocks = sm_count, threads = 128;
    int N = blocks * threads;

    printf("# PDL dep-chain follow-up — B300, %d SMs, %d blocks x %d threads\n\n",
           sm_count, blocks, threads);

    size_t dep_sz = (size_t)N * sizeof(float);
    size_t tile_sz = (size_t)N * 4 * sizeof(float);
    float *buf_A, *buf_B, *buf_tile;
    CK(cudaMalloc(&buf_A, dep_sz));
    CK(cudaMalloc(&buf_B, dep_sz));
    CK(cudaMalloc(&buf_tile, tile_sz));
    CK(cudaMemset(buf_A, 0x3F, dep_sz));
    CK(cudaMemset(buf_B, 0, dep_sz));
    CK(cudaMemset(buf_tile, 0, tile_sz));

    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    cudaLaunchAttribute attr_pss;
    attr_pss.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr_pss.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg_pss = {dim3(blocks), dim3(threads), 0, s, &attr_pss, 1};

    int iters = 5000;
    int sig = (iters * 99) / 100;

    // ========== Test A: Isolate griddepcontrol.wait overhead ==========
    // Compare: full-write kernel WITHOUT wait vs. WITH wait (using PSS)
    // The "first" kernel in chain doesn't wait; subsequent ones do
    // If wait adds overhead, we'll see it here with PSS enabled
    printf("## A: Isolate wait overhead — 1-kernel chains\n");
    printf("#    Single kernel: plain vs. pdl_first (no wait) vs. pdl_mid (has wait)\n");
    {
        float t_plain = bench_fn([&]{
            k_nowait<<<blocks, threads, 0, s>>>(buf_tile, iters);
        }, s, e0, e1);

        float t_first = bench_fn([&]{
            void *a[] = {&buf_tile, &iters, &sig};
            cudaLaunchKernelExC(&cfg_pss, (void*)k_signal_only, a);
        }, s, e0, e1);

        float t_mid = bench_fn([&]{
            void *a[] = {&buf_tile, &iters};
            cudaLaunchKernelExC(&cfg_pss, (void*)k_wait_only, a);
        }, s, e0, e1);

        printf("  plain (no PDL):        %.4f ms\n", t_plain);
        printf("  pdl_first (signal):    %.4f ms (delta=%+.2f us)\n",
               t_first, (t_first - t_plain) * 1000.f);
        printf("  pdl_mid (wait+signal): %.4f ms (delta=%+.2f us)\n",
               t_mid, (t_mid - t_plain) * 1000.f);
    }

    // ========== Test B: 2-kernel chain, isolate wait overhead ==========
    printf("\n## B: 2-kernel chain — measure wait overhead when producer writes\n");
    printf("#    plain → plain  vs  pdl_first(sig) → wait_only\n");
    {
        float t_plain2 = bench_fn([&]{
            k_nowait<<<blocks, threads, 0, s>>>(buf_tile, iters);
            k_nowait<<<blocks, threads, 0, s>>>(buf_tile, iters);
        }, s, e0, e1);

        float t_pdl2 = bench_fn([&]{
            {void *a[] = {&buf_tile, &iters, &sig};
             cudaLaunchKernelExC(&cfg_pss, (void*)k_signal_only, a);}
            void *b[] = {&buf_tile, &iters};
            cudaLaunchKernelExC(&cfg_pss, (void*)k_wait_only, b);
        }, s, e0, e1);

        printf("  2× plain:     %.4f ms (%.2f us/kern)\n", t_plain2, t_plain2 * 500.f);
        printf("  pdl_first+wait: %.4f ms (%.2f us/kern, save=%+.2f us)\n",
               t_pdl2, t_pdl2 * 500.f, (t_plain2 - t_pdl2) * 1000.f);
    }

    // ========== Test C: Dep chain with varying write density ==========
    // Does write SIZE (bytes written) correlate with PDL penalty?
    printf("\n## C: Dep chain, 32-kernel, varying write density (# floats written)\n");
    printf("#    Tests: does griddepcontrol.wait create memory fence proportional to write size?\n");
    printf("# (Using write-tile kernel but only 1-4 floats per thread)\n");
    printf("# writes_per_thread  nopdl_us/k  pdl_us/k  save_us/k\n");

    // Use k_dep (read then write) with only SOME threads writing
    // Approximate by using separate kernels; for simplicity, measure k_dep_plain vs k_dep_pdl
    // but vary whether write is present
    {
        float *src = buf_A, *dst = buf_B;
        for (int n : {32}) {
            for (int itr : {500, 1000, 2500, 5000, 10000}) {
                int sg = (itr * 99) / 100;
                float t_nopdl = bench_fn([&]{
                    float *s_ = src, *d_ = dst;
                    for (int k = 0; k < n; k++) {
                        k_dep_plain<<<blocks, threads, 0, s>>>(s_, d_, itr, N);
                        float *tmp = s_; s_ = d_; d_ = tmp;
                    }
                }, s, e0, e1);

                float t_pss = bench_fn([&]{
                    float *s_ = src, *d_ = dst;
                    {void *a[] = {&s_, &d_, &itr, &N, &sg};
                     cudaLaunchKernelExC(&cfg_pss, (void*)k_dep_pdl_first, a);}
                    float *tmp = s_; s_ = d_; d_ = tmp;
                    for (int k = 1; k < n; k++) {
                        void *a[] = {&s_, &d_, &itr, &N, &sg};
                        cudaLaunchKernelExC(&cfg_pss, (void*)k_dep_pdl_mid, a);
                        tmp = s_; s_ = d_; d_ = tmp;
                    }
                }, s, e0, e1);

                printf("  iters=%-6d  %.2f  %.2f  %+.2f\n",
                       itr, t_nopdl*1000.f/n, t_pss*1000.f/n, (t_nopdl-t_pss)*1000.f/n);
            }
        }
    }

    // ========== Test D: Pure signal overhead (no wait) ==========
    // Does adding PDL launch attribute to a chain (signal only, no wait)
    // save time compared to plain launches?
    printf("\n## D: Pure PSS signal overhead — signal only, no wait, 32-kernel chains\n");
    printf("#    Tests: does launch attribute alone (griddepcontrol.launch_dependents) cost anything\n");
    printf("#    when no kernel waits for it?\n");
    {
        int n = 32;
        for (int itr : {1000, 5000, 25000}) {
            int sg = (itr * 99) / 100;
            float t_plain = bench_fn([&]{
                for (int k = 0; k < n; k++)
                    k_nowait<<<blocks, threads, 0, s>>>(buf_tile, itr);
            }, s, e0, e1);

            // All kernels signal but none wait — first has no wait, all signal
            float t_signal_chain = bench_fn([&]{
                for (int k = 0; k < n; k++) {
                    void *a[] = {&buf_tile, &itr, &sg};
                    cudaLaunchKernelExC(&cfg_pss, (void*)k_signal_only, a);
                }
            }, s, e0, e1);

            printf("  iters=%-6d  plain=%.4f ms (%.2f us/k)  signal_chain=%.4f ms (%.2f us/k)  save=%+.2f us/k\n",
                   itr, t_plain, t_plain*1000.f/n,
                   t_signal_chain, t_signal_chain*1000.f/n,
                   (t_plain - t_signal_chain)*1000.f/n);
        }
    }

    printf("\n# Done.\n");
    CK(cudaFree(buf_A)); CK(cudaFree(buf_B)); CK(cudaFree(buf_tile));
    return 0;
}
