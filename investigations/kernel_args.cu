// Kernel argument size and cost
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

template<int N>
struct BigArgs {
    int values[N];
};

__global__ void k_small(int a, int b, int c) { if (threadIdx.x == 0 && blockIdx.x == 0 && a==99) printf(""); }
__global__ void k_med(BigArgs<16> args) { if (threadIdx.x == 0 && blockIdx.x == 0 && args.values[0]==99) printf(""); }
__global__ void k_big(BigArgs<128> args) { if (threadIdx.x == 0 && blockIdx.x == 0 && args.values[0]==99) printf(""); }
__global__ void k_huge(BigArgs<1023> args) { if (threadIdx.x == 0 && blockIdx.x == 0 && args.values[0]==99) printf(""); }

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials = 200) {
        for (int i = 0; i < 5; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 kernel argument size cost\n\n");

    BigArgs<16> a16; for (int i = 0; i < 16; i++) a16.values[i] = i;
    BigArgs<128> a128; for (int i = 0; i < 128; i++) a128.values[i] = i;
    BigArgs<1023> a1023; for (int i = 0; i < 1023; i++) a1023.values[i] = i;

    printf("  noop kernel (12 B args - 3 ints): %.2f us\n",
           bench([&]{ k_small<<<1, 32, 0, s>>>(1, 2, 3); }));
    printf("  noop kernel (64 B args):          %.2f us\n",
           bench([&]{ k_med<<<1, 32, 0, s>>>(a16); }));
    printf("  noop kernel (512 B args):         %.2f us\n",
           bench([&]{ k_big<<<1, 32, 0, s>>>(a128); }));
    printf("  noop kernel (4092 B args):        %.2f us\n",
           bench([&]{ k_huge<<<1, 32, 0, s>>>(a1023); }));

    // Test launch-only without sync
    printf("\n## Launch latency only (no sync, async submission rate)\n");
    {
        for (int i = 0; i < 100; i++) k_small<<<1, 32, 0, s>>>(1, 2, 3);
        cudaStreamSynchronize(s);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; i++) k_small<<<1, 32, 0, s>>>(1, 2, 3);
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1-t0).count();
        printf("  1000 small launches submission: %.0f us = %.2f us/launch\n", us, us/1000);

        cudaStreamSynchronize(s);
    }

    return 0;
}
