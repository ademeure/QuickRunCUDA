// V6 E4: Per-SM persistent state cost — when does TCB context dump dominate?
// Test: launch persistent kernel with varying register usage + SMEM
// Measure launch-to-first-execution latency
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

template<int N_REGS>
__global__ __launch_bounds__(256, 1)
void persistent_template(unsigned int* done_flag) {
    // Force N_REGS-worth of live registers
    float r[N_REGS];
    #pragma unroll
    for (int i = 0; i < N_REGS; i++) r[i] = (float)(threadIdx.x + i);

    // Set done flag immediately
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *done_flag = 1;
        __threadfence_system();
    }

    // Anti-DCE
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < N_REGS; i++) sum += r[i];
    if (sum == 1.234567e-30f && threadIdx.x == 0) done_flag[1] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* done_flag;
    cudaHostAlloc(&done_flag, 16 * sizeof(unsigned int), cudaHostAllocMapped);
    unsigned int* dev_done;
    cudaHostGetDevicePointer(&dev_done, done_flag, 0);

    int N_RUNS = 100;

    // Warmup
    persistent_template<4><<<1, 256>>>(dev_done);
    cudaDeviceSynchronize();

    // Test various register pressures
    const char* labels[] = {"4 regs", "16 regs", "64 regs", "128 regs"};
    for (int mode = 0; mode < 4; mode++) {
        *done_flag = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_RUNS; i++) {
            *done_flag = 0;
            switch (mode) {
                case 0: persistent_template<4><<<1, 256>>>(dev_done); break;
                case 1: persistent_template<16><<<1, 256>>>(dev_done); break;
                case 2: persistent_template<64><<<1, 256>>>(dev_done); break;
                case 3: persistent_template<128><<<1, 256>>>(dev_done); break;
            }
            while (__atomic_load_n(done_flag, __ATOMIC_ACQUIRE) != 1) {}
            cudaStreamSynchronize(0);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double per_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

        // Query register usage
        cudaFuncAttributes attr;
        switch (mode) {
            case 0: cudaFuncGetAttributes(&attr, persistent_template<4>); break;
            case 1: cudaFuncGetAttributes(&attr, persistent_template<16>); break;
            case 2: cudaFuncGetAttributes(&attr, persistent_template<64>); break;
            case 3: cudaFuncGetAttributes(&attr, persistent_template<128>); break;
        }
        printf("%s: launch+exec RTT = %.2f us  (actual regs=%d)\n",
               labels[mode], per_us, attr.numRegs);
    }

    return 0;
}
