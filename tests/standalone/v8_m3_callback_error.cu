// V8 M3: Stream callback safety under kernel error
// Does cudaLaunchHostFunc still fire if a previous kernel in stream errors?
#include <cuda_runtime.h>
#include <cstdio>
#include <atomic>

std::atomic<int> cb_fired(0);

void CUDART_CB callback(void* data) {
    cb_fired.fetch_add(1, std::memory_order_relaxed);
}

__global__ void crash(unsigned int* buf) {
    // Deliberately access invalid memory
    if (threadIdx.x == 0) buf[1000000000] = 1;  // way out of bounds
}

__global__ void noop(unsigned int* buf) {
    if (threadIdx.x == 0) buf[0] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMalloc(&buf, 4);
    cudaStream_t s; cudaStreamCreate(&s);

    // Test 1: callback after successful kernel
    noop<<<1, 32, 0, s>>>(buf);
    cudaLaunchHostFunc(s, callback, nullptr);
    cudaStreamSynchronize(s);
    int after_noop = cb_fired.load();
    printf("Callback after noop: fired=%d (expect 1)\n", after_noop);

    // Test 2: callback after crashing kernel
    cb_fired.store(0);
    crash<<<1, 32, 0, s>>>(buf);
    cudaLaunchHostFunc(s, callback, nullptr);
    cudaError_t err = cudaStreamSynchronize(s);
    int after_crash = cb_fired.load();
    printf("Callback after crash: fired=%d (sync err=%s)\n", after_crash, cudaGetErrorString(err));

    // Test 3: launch noop after error — does context recover?
    cudaError_t reset_err = cudaDeviceReset();
    printf("After cudaDeviceReset: %s\n", cudaGetErrorString(reset_err));

    return 0;
}
