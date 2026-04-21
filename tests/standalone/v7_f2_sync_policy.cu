// V7 F2: Stream attribute cudaSyncPolicy
// MODE 0: Auto (default)
// MODE 1: Spin (busy-wait, low latency)
// MODE 2: Yield (yield CPU thread)
// MODE 3: BlockingSync (sleep)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void short_kernel(unsigned int* buf) {
    if (threadIdx.x == 0) buf[0] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4);

    int N_RUNS = 1000;
    const char* names[] = {"Auto", "Spin", "Yield", "BlockingSync"};
    cudaSynchronizationPolicy policies[] = {
        cudaSyncPolicyAuto, cudaSyncPolicySpin, cudaSyncPolicyYield, cudaSyncPolicyBlockingSync
    };

    for (int mode = 0; mode < 4; mode++) {
        cudaStream_t s;
        cudaStreamCreate(&s);

        cudaStreamAttrValue attr = {};
        attr.syncPolicy = policies[mode];
        cudaStreamSetAttribute(s, cudaStreamAttributeSynchronizationPolicy, &attr);

        // Warmup
        short_kernel<<<1, 32, 0, s>>>(buf);
        cudaStreamSynchronize(s);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_RUNS; i++) {
            short_kernel<<<1, 32, 0, s>>>(buf);
            cudaStreamSynchronize(s);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double per_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;
        printf("SyncPolicy=%-13s launch+sync: %.2f us\n", names[mode], per_us);

        cudaStreamDestroy(s);
    }

    return 0;
}
