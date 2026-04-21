// J1: NVLink raw round-trip latency
// GPU 0: write flag → wait for response from GPU 1
// GPU 1: spin-read flag from GPU 0, then write response
// Measure round-trip time
#include <cuda_runtime.h>
#include <cstdio>

__global__ void ping(volatile unsigned int* my_buf, volatile unsigned int* peer_buf,
                     unsigned long long* time_out, int iters) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 1; i <= iters; i++) {
        // Send: write to peer (across NVLink)
        *peer_buf = (unsigned int)i;
        // Wait: spin until peer increments my_buf to i+0x100000
        while (*my_buf != (unsigned int)(i + 0x100000)) { /* spin */ }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    *time_out = t1 - t0;
}

__global__ void pong(volatile unsigned int* my_buf, volatile unsigned int* peer_buf, int iters) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int i = 1; i <= iters; i++) {
        // Wait: spin until ping writes i to my_buf
        while (*my_buf != (unsigned int)i) { /* spin */ }
        // Reply: write i+0x100000 to peer
        *peer_buf = (unsigned int)(i + 0x100000);
    }
}

int main() {
    int iters = 1000;

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    unsigned int* gpu0_buf;
    unsigned long long* gpu0_time;
    cudaMalloc(&gpu0_buf, sizeof(unsigned int));
    cudaMalloc(&gpu0_time, sizeof(unsigned long long));
    cudaMemset(gpu0_buf, 0, sizeof(unsigned int));

    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);
    unsigned int* gpu1_buf;
    cudaMalloc(&gpu1_buf, sizeof(unsigned int));
    cudaMemset(gpu1_buf, 0, sizeof(unsigned int));

    cudaStream_t s0, s1;
    cudaSetDevice(0);
    cudaStreamCreate(&s0);
    cudaSetDevice(1);
    cudaStreamCreate(&s1);

    // Launch pong on GPU 1 first (it needs to spin)
    cudaSetDevice(1);
    pong<<<1, 32, 0, s1>>>(gpu1_buf, gpu0_buf, iters);

    // Launch ping on GPU 0
    cudaSetDevice(0);
    ping<<<1, 32, 0, s0>>>(gpu0_buf, gpu1_buf, gpu0_time, iters);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    // Read time
    unsigned long long total_cy;
    cudaSetDevice(0);
    cudaMemcpy(&total_cy, gpu0_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Convert cy to seconds (B300 clock-locked at 1500 MHz)
    double cy_per_iter = (double)total_cy / iters;
    double ns_per_iter = cy_per_iter / 1.5;  // 1500 MHz = 1.5 cy/ns
    printf("NVLink ping-pong: %d iterations\n", iters);
    printf("  total cycles (gpu0): %llu\n", total_cy);
    printf("  cy/round-trip:       %.1f\n", cy_per_iter);
    printf("  ns/round-trip:       %.1f\n", ns_per_iter);
    printf("  one-way latency:     %.1f ns\n", ns_per_iter / 2);

    return 0;
}
