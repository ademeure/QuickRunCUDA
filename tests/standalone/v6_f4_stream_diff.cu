// V6 F4: Default stream vs custom stream overhead
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop(unsigned int* buf) {
    if (threadIdx.x == 0 && blockIdx.x == 0) buf[0] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4);

    int N_RUNS = 1000;

    // Warmup
    noop<<<1, 32>>>(buf);
    cudaDeviceSynchronize();

    // Default stream (legacy nullstream)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) noop<<<1, 32>>>(buf);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double default_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    // Per-thread default stream — use cudaStreamPerThread directly (it's a special handle, not a flag)
    cudaStream_t s_perthr = cudaStreamPerThread;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) noop<<<1, 32, 0, s_perthr>>>(buf);
    cudaStreamSynchronize(s_perthr);
    auto t3 = std::chrono::high_resolution_clock::now();
    double perthr_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;

    // Non-blocking custom stream
    cudaStream_t s_nb;
    cudaStreamCreateWithFlags(&s_nb, cudaStreamNonBlocking);
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) noop<<<1, 32, 0, s_nb>>>(buf);
    cudaStreamSynchronize(s_nb);
    auto t5 = std::chrono::high_resolution_clock::now();
    double nb_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / N_RUNS;

    // Regular custom stream
    cudaStream_t s_reg;
    cudaStreamCreate(&s_reg);
    auto t6 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) noop<<<1, 32, 0, s_reg>>>(buf);
    cudaStreamSynchronize(s_reg);
    auto t7 = std::chrono::high_resolution_clock::now();
    double reg_us = std::chrono::duration<double, std::micro>(t7 - t6).count() / N_RUNS;

    printf("Default stream (nullstream):   %.2f us/launch\n", default_us);
    printf("cudaStreamPerThread:           %.2f us/launch\n", perthr_us);
    printf("Custom non-blocking:           %.2f us/launch\n", nb_us);
    printf("Custom regular:                %.2f us/launch\n", reg_us);

    return 0;
}
