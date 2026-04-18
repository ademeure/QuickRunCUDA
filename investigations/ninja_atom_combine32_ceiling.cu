// Push int32 combine=32 atomic to its true ceiling
#include <cuda_runtime.h>
#include <cstdio>

template <int OCC>
__launch_bounds__(256, OCC) __global__ void atom_combine32(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_warps = ((long)gridDim.x * blockDim.x) / 32;
    int warp_id = tid / 32;
    int lane = tid & 31;
    for (int i = 0; i < N_iters; i++) {
        long base = (warp_id + (long)i * N_warps) * 32;
        atomicAdd(&p[(base + lane) % N_addrs], 1);
    }
}

template <int OCC>
double bench(int *d_p, long N_addrs, int N_iters, const char* label) {
    int blocks = 148 * OCC;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) atom_combine32<OCC><<<blocks, 256>>>(d_p, N_iters, N_addrs);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR\n"); return 0; }
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        atom_combine32<OCC><<<blocks, 256>>>(d_p, N_iters, N_addrs);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = (long)blocks * 256 * N_iters;
    double gops = ops / (best/1000.0) / 1e9;
    long packets = ops / 32;
    double l2_packet_rate = packets / (best/1000.0) / 1e9;
    double payload = ops * 4.0 / (best/1000.0) / 1e9;
    printf("  OCC=%d (%dthr/SM, %d blocks)  %s  %.1f Gops payload=%.0f GB/s  L2 packets=%.1f G/s\n",
        OCC, OCC*256, blocks, label, gops, payload, l2_packet_rate);
    return gops;
}

int main(int argc, char** argv) {
    cudaSetDevice(0);
    long WS_MB = (argc > 1) ? atol(argv[1]) : 1024;
    long N = WS_MB * 1024 * 1024 / 4;
    int *d_p; cudaMalloc(&d_p, N * 4); cudaMemset(d_p, 0, N * 4);
    int N_iters = 100;
    printf("# combine=32 atomic, WS=%ldMB\n", WS_MB);
    bench<1>(d_p, N, N_iters, "");
    bench<2>(d_p, N, N_iters, "");
    bench<4>(d_p, N, N_iters, "");
    bench<8>(d_p, N, N_iters, "");
    return 0;
}
