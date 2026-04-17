// CUDA Streams: real concurrent execution with proper cross-stream timing
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void compute(float *out, int iters, int sentinel) {
    float a = 1.0f + (threadIdx.x & 31) * 0.001f + sentinel * 0.00001f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

    printf("# B300 Streams REAL concurrent test (CPU-side timing with sync)\n");
    int prio_low, prio_high;
    CK(cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high));
    printf("# Priority range: %d (high) to %d (low) = %d levels\n",
           prio_high, prio_low, prio_low - prio_high + 1);

    float *d_out;
    CK(cudaMalloc(&d_out, 1024 * sizeof(float)));
    CK(cudaMemset(d_out, 0, 1024 * sizeof(float)));

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

    int blocks = 1, threads = 128, iters = 50000;

    float t_one = bench_sync([&]{
        compute<<<blocks, threads, 0, 0>>>(d_out, iters, 0);
    });
    printf("\n# Single kernel (1 block, 128 threads, %d iters): %.4f ms\n", iters, t_one);

    // ===== Multi-stream parallel concurrent =====
    printf("\n## N kernels parallel on N streams (%d block, %d threads each, %d iters):\n",
           blocks, threads, iters);
    printf("# %-10s %-12s %-12s %-12s\n", "n_streams", "time_ms", "vs_single", "concurrency");

    int n_arr[] = {1, 2, 4, 8, 16, 32, 64, 128, 148, 192, 256, 512};
    for (int ni = 0; ni < 12; ni++) {
        int n = n_arr[ni];
        std::vector<cudaStream_t> ss(n);
        for (int i = 0; i < n; i++) CK(cudaStreamCreateWithFlags(&ss[i], cudaStreamNonBlocking));

        float t = bench_sync([&]{
            for (int i = 0; i < n; i++)
                compute<<<blocks,threads,0,ss[i]>>>(d_out, iters, i);
        });

        // Concurrency: how many ran in parallel = (n * t_one) / t_actual
        float concurrency = (n * t_one) / t;
        printf("  %-10d %-12.4f %-12.2fx %.2fx parallel\n", n, t, t / t_one, concurrency);

        for (auto &s : ss) cudaStreamDestroy(s);
    }

    // ===== Stream priority preemption =====
    printf("\n## Stream priority effects (full-GPU kernels)\n");
    {
        int n_blocks_full = sm_count;
        cudaStream_t s_low, s_high;
        CK(cudaStreamCreateWithPriority(&s_low, cudaStreamNonBlocking, prio_low));
        CK(cudaStreamCreateWithPriority(&s_high, cudaStreamNonBlocking, prio_high));

        float t_low_alone = bench_sync([&]{
            compute<<<n_blocks_full, threads, 0, s_low>>>(d_out, iters, 1);
        });
        float t_high_alone = bench_sync([&]{
            compute<<<n_blocks_full, threads, 0, s_high>>>(d_out, iters, 2);
        });

        // Both filling GPU concurrently
        float t_par = bench_sync([&]{
            compute<<<n_blocks_full, threads, 0, s_low>>>(d_out, iters, 1);
            compute<<<n_blocks_full, threads, 0, s_high>>>(d_out, iters, 2);
        });

        // Reverse order
        float t_par_hl = bench_sync([&]{
            compute<<<n_blocks_full, threads, 0, s_high>>>(d_out, iters, 2);
            compute<<<n_blocks_full, threads, 0, s_low>>>(d_out, iters, 1);
        });

        printf("  low alone:           %.4f ms\n", t_low_alone);
        printf("  high alone:          %.4f ms\n", t_high_alone);
        printf("  low|high parallel:   %.4f ms (sum = %.4f, ratio = %.2fx)\n",
               t_par, t_low_alone + t_high_alone, t_par / (t_low_alone + t_high_alone));
        printf("  high|low parallel:   %.4f ms (ratio = %.2fx)\n",
               t_par_hl, t_par_hl / (t_low_alone + t_high_alone));

        // Half-GPU each, both running parallel — should fit
        int n_blocks_half = sm_count / 2;
        float t_half_alone = bench_sync([&]{
            compute<<<n_blocks_half, threads, 0, s_low>>>(d_out, iters, 1);
        });
        float t_half_par = bench_sync([&]{
            compute<<<n_blocks_half, threads, 0, s_low>>>(d_out, iters, 1);
            compute<<<n_blocks_half, threads, 0, s_high>>>(d_out, iters, 2);
        });
        printf("\n  Half-GPU each (74 blocks):\n");
        printf("    half alone:        %.4f ms\n", t_half_alone);
        printf("    half|half parallel: %.4f ms (ratio %.2fx, ideal=1.0)\n",
               t_half_par, t_half_par / t_half_alone);

        cudaStreamDestroy(s_low);
        cudaStreamDestroy(s_high);
    }

    // ===== Two streams: scaling block count =====
    printf("\n## Two streams parallel, varying block count per stream\n");
    printf("# %-10s %-12s %-12s %-12s\n", "blocks/str", "single_ms", "par_ms", "par_ratio");
    {
        cudaStream_t s1, s2;
        CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

        int b_arr[] = {1, 2, 4, 8, 16, 32, 64, 74, 100, 148, 296};
        for (int bi = 0; bi < 11; bi++) {
            int b = b_arr[bi];
            float t_single = bench_sync([&]{
                compute<<<b, threads, 0, s1>>>(d_out, iters, 5);
            });
            float t_par = bench_sync([&]{
                compute<<<b, threads, 0, s1>>>(d_out, iters, 5);
                compute<<<b, threads, 0, s2>>>(d_out, iters, 6);
            });
            printf("  %-10d %-12.4f %-12.4f %.2fx (ideal: 1.0 if fits, 2.0 if not)\n",
                   b, t_single, t_par, t_par / t_single);
        }

        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
    }

    CK(cudaFree(d_out));
    return 0;
}
