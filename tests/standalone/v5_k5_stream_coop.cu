// V5 K5: Stream-ordered cooperation patterns
// Compare:
//   1. Sequential (1 stream, 2 kernels)
//   2. Parallel (2 streams, 2 independent kernels)
//   3. Dependent (2 streams w/ event waits)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void work(unsigned int* buf, int idx, int delay_iters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < (unsigned long long)delay_iters);
        buf[idx] = (unsigned)idx;
    }
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    cudaEvent_t evt;
    cudaEventCreate(&evt);

    unsigned int* buf;
    cudaMallocManaged(&buf, 1024);

    int delay = 1500000;  // ~1 ms each kernel
    int N = 100;

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);

    // Warmup
    work<<<1, 32, 0, s1>>>(buf, 0, delay);
    work<<<1, 32, 0, s2>>>(buf, 1, delay);
    cudaDeviceSynchronize();

    // Test 1: SEQUENTIAL — same stream, 2 kernels
    cudaEventRecord(es, s1);
    for (int i = 0; i < N; i++) {
        work<<<1, 32, 0, s1>>>(buf, 0, delay);
        work<<<1, 32, 0, s1>>>(buf, 1, delay);
    }
    cudaEventRecord(ee, s1);
    cudaEventSynchronize(ee);
    float seq_ms;
    cudaEventElapsedTime(&seq_ms, es, ee);
    printf("Sequential (1 stream, 2 serialized kernels per iter): %.3f ms total = %.3f us per pair\n",
           seq_ms, seq_ms * 1000 / N);

    // Test 2: PARALLEL — 2 streams independent
    cudaEventRecord(es, 0);
    for (int i = 0; i < N; i++) {
        work<<<1, 32, 0, s1>>>(buf, 0, delay);
        work<<<1, 32, 0, s2>>>(buf, 1, delay);
    }
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    cudaEventRecord(ee, 0);
    cudaEventSynchronize(ee);
    float par_ms;
    cudaEventElapsedTime(&par_ms, es, ee);
    printf("Parallel (2 streams, independent): %.3f ms total = %.3f us per pair\n",
           par_ms, par_ms * 1000 / N);

    // Test 3: DEPENDENT — s2 waits on s1's event
    cudaEventRecord(es, 0);
    for (int i = 0; i < N; i++) {
        work<<<1, 32, 0, s1>>>(buf, 0, delay);
        cudaEventRecord(evt, s1);
        cudaStreamWaitEvent(s2, evt, 0);
        work<<<1, 32, 0, s2>>>(buf, 1, delay);
    }
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    cudaEventRecord(ee, 0);
    cudaEventSynchronize(ee);
    float dep_ms;
    cudaEventElapsedTime(&dep_ms, es, ee);
    printf("Dependent (s2 waits s1 event): %.3f ms total = %.3f us per pair\n",
           dep_ms, dep_ms * 1000 / N);

    return 0;
}
