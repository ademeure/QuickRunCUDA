// F2 RIGOR: mbarrier R/W bandwidth via phase parity tight loop
//
// THEORETICAL: mbarrier ops are SHMEM-resident barriers. Each op = 1 shared
// memory access (atomic). Per-SM SHMEM peak ~260 GB/s, atomic 4-cycle issue
// suggests ~50-100 Gops/s of mbarrier ops per SM.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

#ifndef ITERS
#define ITERS 1024
#endif

// Tight loop: arrive then test_wait checking phase parity flip
extern "C" __launch_bounds__(256, 4) __global__ void mbar_arrive_wait(uint64_t *out_cycles) {
    __shared__ uint64_t mbar;
    __shared__ uint64_t arrive_token;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "l"(&mbar), "r"((unsigned)blockDim.x));
    }
    __syncthreads();

    unsigned long long t0 = clock64();
    unsigned phase = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
            :: "l"(&mbar)
        );
        // Use try_wait.parity which loops on a phase parity (0 or 1)
        unsigned ready = 0;
        do {
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                "selp.u32 %0, 1, 0, p;\n"
                "}\n"
                : "=r"(ready) : "l"(&mbar), "r"(phase));
        } while (!ready);
        phase ^= 1;
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out_cycles[blockIdx.x] = t1 - t0;
}

// Measure plain shared-memory atomicAdd as comparison
extern "C" __launch_bounds__(256, 4) __global__ void smem_atomic_chain(uint64_t *out_cycles) {
    __shared__ unsigned smem_counter;
    if (threadIdx.x == 0) smem_counter = 0;
    __syncthreads();

    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        atomicAdd(&smem_counter, 1);
        __syncthreads();
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out_cycles[blockIdx.x] = t1 - t0;
}

int main() {
    cudaSetDevice(0);
    int blocks = 148;
    uint64_t *d_cycles; cudaMalloc(&d_cycles, blocks * sizeof(uint64_t));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch, const char* label) {
        for (int i = 0; i < 5; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 10; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        std::vector<uint64_t> cycles(blocks);
        cudaMemcpy(cycles.data(), d_cycles, blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        uint64_t min_c = cycles[0], max_c = cycles[0], sum_c = 0;
        for (auto c : cycles) { min_c = std::min(min_c, c); max_c = std::max(max_c, c); sum_c += c; }
        double avg_c = sum_c / (double)blocks;
        printf("  %s: %.4f ms wall = %.1f ns/op (per block %d)\n",
               label, best, best * 1e6 / ITERS, blocks);
        printf("    clock64: avg %.1f cy/op (= %.1f ns @ 1920 MHz)\n",
               avg_c / ITERS, avg_c / ITERS / 1.92);
    };

    bench([&]{ mbar_arrive_wait<<<blocks, 256>>>(d_cycles); }, "mbar arrive+test_wait");
    bench([&]{ smem_atomic_chain<<<blocks, 256>>>(d_cycles); }, "smem atomic+sync     ");

    return 0;
}
