// IPC server: alloc buffer, write handle to a file, hold buffer
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <signal.h>
#include <chrono>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

volatile int keep = 1;
void sigh(int) { keep = 0; }

__global__ void poll_step(volatile int *p, int trig, int next, int max_iter) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < max_iter; i++) {
            while (atomicAdd((int*)p, 0) != trig + 2*i) {}
            __threadfence_system();
            atomicExch((int*)p, next + 2*i);
        }
    }
}

int main(int argc, char** argv) {
    cudaSetDevice(0);
    size_t bytes = 64 * 1024;
    void *d_buf;
    CHK(cudaMalloc(&d_buf, bytes));
    cudaMemset(d_buf, 0, bytes);
    cudaDeviceSynchronize();

    cudaIpcMemHandle_t handle;

    // Time GetMemHandle
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        CHK(cudaIpcGetMemHandle(&handle, d_buf));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double get_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;
    printf("[SRV] cudaIpcGetMemHandle = %.2f us avg (100 trials)\n", get_us);
    fflush(stdout);

    // Write handle to file
    FILE* f = fopen("/tmp/ipc_handle.bin", "wb");
    fwrite(&handle, sizeof(handle), 1, f);
    fclose(f);

    // Tell client we're ready
    f = fopen("/tmp/ipc_srv_ready", "w"); fputs("y", f); fclose(f);

    signal(SIGTERM, sigh);
    signal(SIGINT, sigh);

    // Wait for client to signal ready
    while (keep && access("/tmp/ipc_cli_ready", F_OK) != 0) usleep(1000);

    // Run ping-pong: server starts at 0, on each round increments by 2
    // Server sets val to 1; waits for 2; sets 3; waits for 4; ...
    int N = 1000;
    int *d_flag = (int*)d_buf;
    cudaMemset(d_flag, 0, sizeof(int));
    cudaDeviceSynchronize();

    // Sync: file flag
    f = fopen("/tmp/ipc_pingpong_go", "w"); fputs("y", f); fclose(f);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    // Server: trigger=0, set=1, then trigger=2, set=3, ...
    poll_step<<<1,1>>>(d_flag, 0, 1, N);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    double per_round = ms * 1000.0 / N;
    printf("[SRV] ping-pong %d rounds (server-half), %.4f ms = %.2f us/round\n", N, ms, per_round);

    // Wait for client done
    while (keep && access("/tmp/ipc_cli_done", F_OK) != 0) usleep(1000);

    cudaFree(d_buf);
    unlink("/tmp/ipc_handle.bin");
    unlink("/tmp/ipc_srv_ready");
    unlink("/tmp/ipc_cli_ready");
    unlink("/tmp/ipc_cli_done");
    unlink("/tmp/ipc_pingpong_go");
    return 0;
}
