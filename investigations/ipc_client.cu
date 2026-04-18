// IPC client: read handle, open, measure latencies
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <chrono>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void poll_step(volatile int *p, int trig, int next, int max_iter) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < max_iter; i++) {
            while (atomicAdd((int*)p, 0) != trig + 2*i) {}
            __threadfence_system();
            atomicExch((int*)p, next + 2*i);
        }
    }
}

int main() {
    cudaSetDevice(0);

    // Wait for server ready
    while (access("/tmp/ipc_srv_ready", F_OK) != 0) usleep(1000);

    // Read handle
    cudaIpcMemHandle_t handle;
    FILE* f = fopen("/tmp/ipc_handle.bin", "rb");
    fread(&handle, sizeof(handle), 1, f);
    fclose(f);

    // Time OpenMemHandle (single shot)
    void *d_buf;
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaError_t err = cudaIpcOpenMemHandle(&d_buf, handle, cudaIpcMemLazyEnablePeerAccess);
    auto t1 = std::chrono::high_resolution_clock::now();
    double open_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CLI] cudaIpcOpenMemHandle failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("[CLI] cudaIpcOpenMemHandle = %.2f us (single shot)\n", open_us);

    // Time first cold read (D2H 4 bytes)
    int h = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(&h, d_buf, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double first_read_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    printf("[CLI] first read D2H 4B = %.2f us (val=%d)\n", first_read_us, h);

    // Sustained: 1000 small D2H reads
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 1000; i++) {
        cudaMemcpyAsync(&h, d_buf, sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    printf("[CLI] sustained D2H 4B x1000 = %.4f ms total = %.2f us/op\n", ms, ms * 1000.0 / 1000.0);

    // Sustained: 1000 large D2H copies (64 KB)
    char *h_buf = (char*)malloc(64 * 1024);
    cudaEventRecord(e0);
    for (int i = 0; i < 100; i++) {
        cudaMemcpyAsync(h_buf, d_buf, 64 * 1024, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    printf("[CLI] sustained D2H 64KB x100 = %.4f ms = %.2f us/op = %.2f GB/s\n",
           ms, ms * 1000.0 / 100.0, (64.0 * 1024 * 100) / (ms / 1000.0) / 1e9);

    // Signal ready for ping-pong
    f = fopen("/tmp/ipc_cli_ready", "w"); fputs("y", f); fclose(f);

    // Wait for go
    while (access("/tmp/ipc_pingpong_go", F_OK) != 0) usleep(1000);

    // Ping-pong: client waits for 1, sets 2; waits for 3, sets 4...
    int N = 1000;
    int *d_flag = (int*)d_buf;
    cudaEventRecord(e0);
    poll_step<<<1, 1>>>(d_flag, 1, 2, N);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    double per_round = ms * 1000.0 / N;
    // half-round = client waiting for one transition + setting one. Full round is 2 transitions.
    printf("[CLI] ping-pong %d rounds (client-half), %.4f ms = %.2f us/round\n", N, ms, per_round);

    cudaIpcCloseMemHandle(d_buf);
    free(h_buf);

    f = fopen("/tmp/ipc_cli_done", "w"); fputs("y", f); fclose(f);
    return 0;
}
