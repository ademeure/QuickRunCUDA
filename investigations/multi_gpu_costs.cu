// Multi-GPU operation costs on 2× B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    int n_gpus;
    cudaGetDeviceCount(&n_gpus);
    printf("# B300 multi-GPU operation costs (%d GPUs)\n\n", n_gpus);

    if (n_gpus < 2) { printf("Need 2+ GPUs\n"); return 1; }

    auto measure = [](auto fn, int N=1000) {
        for (int i = 0; i < 10; i++) fn();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::micro>(t1-t0).count() / N;
    };

    // Test 1: cudaSetDevice cost (warm)
    cudaSetDevice(0);
    cudaSetDevice(1);  // warm up
    float t_set_warm_same = measure([&]{ cudaSetDevice(0); });
    cudaSetDevice(0);
    float t_set_warm_switch = measure([&]{
        cudaSetDevice(1); cudaSetDevice(0);
    });
    printf("  cudaSetDevice(0) when already 0:     %.3f us\n", t_set_warm_same);
    printf("  cudaSetDevice toggle (1→0→1→0):     %.3f us per call\n", t_set_warm_switch / 2);

    // Test 2: cudaDeviceCanAccessPeer
    int can_p2p;
    float t_can_peer = measure([&]{ cudaDeviceCanAccessPeer(&can_p2p, 0, 1); });
    printf("  cudaDeviceCanAccessPeer:              %.3f us\n", t_can_peer);

    // Test 3: cudaDeviceEnable/DisablePeerAccess
    cudaSetDevice(0);
    cudaDeviceDisablePeerAccess(1);  // ensure clean state
    cudaGetLastError();

    auto t0 = std::chrono::high_resolution_clock::now();
    cudaDeviceEnablePeerAccess(1, 0);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("  cudaDeviceEnablePeerAccess (cold):    %.3f us\n",
           std::chrono::duration<float, std::micro>(t1-t0).count());

    t0 = std::chrono::high_resolution_clock::now();
    cudaDeviceDisablePeerAccess(1);
    t1 = std::chrono::high_resolution_clock::now();
    printf("  cudaDeviceDisablePeerAccess:          %.3f us\n",
           std::chrono::duration<float, std::micro>(t1-t0).count());

    // Test 4: Stream switching across devices
    cudaSetDevice(0);
    cudaStream_t s0; cudaStreamCreate(&s0);
    cudaSetDevice(1);
    cudaStream_t s1; cudaStreamCreate(&s1);

    float t_event_xdev = measure([&]{
        cudaEvent_t e;
        cudaEventCreate(&e);
        cudaEventRecord(e, s0);
        cudaStreamWaitEvent(s1, e, 0);
        cudaEventDestroy(e);
    });
    printf("\n  Event cross-device sync setup:        %.3f us\n", t_event_xdev);

    // Test 5: cudaIpcGetMemHandle / OpenMemHandle (cross-process IPC)
    cudaSetDevice(0);
    void *d_ptr;
    cudaMalloc(&d_ptr, 1024 * 1024);
    cudaIpcMemHandle_t handle;

    float t_ipc_get = measure([&]{
        cudaIpcGetMemHandle(&handle, d_ptr);
    }, 100);
    printf("  cudaIpcGetMemHandle:                  %.3f us\n", t_ipc_get);

    // Open within same process (technically in same context, may not work)
    cudaSetDevice(1);
    void *p2;
    cudaError_t r = cudaIpcOpenMemHandle(&p2, handle, cudaIpcMemLazyEnablePeerAccess);
    printf("  cudaIpcOpenMemHandle (same proc):     %s\n",
           r == cudaSuccess ? "OK" : cudaGetErrorString(r));

    cudaSetDevice(0);
    cudaFree(d_ptr);
    cudaStreamDestroy(s0);
    cudaSetDevice(1);
    cudaStreamDestroy(s1);

    return 0;
}
