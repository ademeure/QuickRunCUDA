// Measure various sync primitive costs in detail
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

int main() {
    CK(cudaSetDevice(0));

    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t evt; CK(cudaEventCreate(&evt));

    const int N = 10000;

    auto micro_per = [](auto fn, int n) {
        for (int i = 0; i < 100; i++) fn();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::micro>(t1-t0).count() / n;
    };

    printf("# CUDA sync/event/stream API costs (per call, no kernel work pending)\n\n");

    printf("  cudaDeviceSynchronize (idle):       %.3f us\n",
           micro_per([&]{ cudaDeviceSynchronize(); }, N));

    printf("  cudaStreamSynchronize (idle):       %.3f us\n",
           micro_per([&]{ cudaStreamSynchronize(s); }, N));

    printf("  cudaStreamQuery (idle):             %.3f us\n",
           micro_per([&]{ cudaStreamQuery(s); }, N));

    printf("  cudaEventQuery (signaled):          %.3f us\n",
           micro_per([&]{ cudaEventRecord(evt, s); cudaEventSynchronize(evt); cudaEventQuery(evt); }, N/4));

    printf("  cudaEventRecord:                    %.3f us\n",
           micro_per([&]{ cudaEventRecord(evt, s); }, N));

    printf("  cudaEventRecord + Synchronize:      %.3f us\n",
           micro_per([&]{ cudaEventRecord(evt, s); cudaEventSynchronize(evt); }, N));

    printf("  cudaStreamWaitEvent:                %.3f us\n",
           micro_per([&]{ cudaStreamWaitEvent(s, evt, 0); }, N));

    printf("  cudaEventCreate + Destroy:          %.3f us\n",
           micro_per([&]{
               cudaEvent_t te; cudaEventCreate(&te); cudaEventDestroy(te);
           }, N));

    printf("  cudaEventCreateWithFlags(disable_t) + Destroy: %.3f us\n",
           micro_per([&]{
               cudaEvent_t te; cudaEventCreateWithFlags(&te, cudaEventDisableTiming);
               cudaEventDestroy(te);
           }, N));

    printf("  cudaStreamCreate + Destroy:         %.3f us\n",
           micro_per([&]{
               cudaStream_t ts; cudaStreamCreate(&ts); cudaStreamDestroy(ts);
           }, 1000));

    printf("  cudaStreamCreateWithFlags(NB) + Destroy: %.3f us\n",
           micro_per([&]{
               cudaStream_t ts; cudaStreamCreateWithFlags(&ts, cudaStreamNonBlocking);
               cudaStreamDestroy(ts);
           }, 1000));

    // Stream priority creation
    printf("  cudaStreamCreateWithPriority + Destroy:  %.3f us\n",
           micro_per([&]{
               cudaStream_t ts; cudaStreamCreateWithPriority(&ts, cudaStreamNonBlocking, -3);
               cudaStreamDestroy(ts);
           }, 1000));

    // Empty kernel launch
    printf("\n# Kernel launch:\n");
    auto empty_kern = [&]{ };
    // We need a real kernel for launch test. Skip or use existing.

    cudaStreamDestroy(s);
    cudaEventDestroy(evt);
    return 0;
}
