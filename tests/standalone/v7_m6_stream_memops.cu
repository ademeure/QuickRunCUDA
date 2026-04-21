// V7 M6: cuStreamWriteValue / cuStreamWaitValue overhead
// Compare to direct kernel signal (V6 E1 = 2.77 us)
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cuInit(0);
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);

    unsigned int* sig;
    cudaMallocManaged((void**)&sig, sizeof(unsigned int));
    *sig = 0;
    CUdeviceptr dev_sig = (CUdeviceptr)sig;

    cudaStream_t s;
    cudaStreamCreate(&s);

    int N_RUNS = 1000;

    // Warmup
    cuStreamWriteValue32(s, dev_sig, 1, 0);
    cudaStreamSynchronize(s);

    // Test cuStreamWriteValue (write to memory from stream)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        cuStreamWriteValue32(s, dev_sig, (unsigned int)i, 0);
    }
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double write_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    // Test cuStreamWaitValue (wait until memory matches)
    *sig = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        // Pre-set the value, then wait shouldn't actually wait
        cuStreamWriteValue32(s, dev_sig, (unsigned int)(i + 1000), 0);
        cuStreamWaitValue32(s, dev_sig, (unsigned int)(i + 1000), CU_STREAM_WAIT_VALUE_EQ);
    }
    cudaStreamSynchronize(s);
    auto t3 = std::chrono::high_resolution_clock::now();
    double pair_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;

    printf("cuStreamWriteValue32:           %.2f us/op\n", write_us);
    printf("cuStreamWriteValue+WaitValue:   %.2f us/op (per pair)\n", pair_us);
    printf("Wait-only cost (pair - write):  %.2f us\n", pair_us - write_us);

    return 0;
}
