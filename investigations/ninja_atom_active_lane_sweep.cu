// Sweep N_ACTIVE lanes targeting consecutive 8B addrs, all warps target same N*8 bytes
#include <cuda_runtime.h>
#include <cstdio>

template <int N_ACTIVE>
__launch_bounds__(256, 8) __global__ void exch64_n(unsigned long long *p, int N_iters) {
    int lane = threadIdx.x & 31;
    if (lane >= N_ACTIVE) return;
    unsigned long long v = (unsigned long long)threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        v = atomicExch(&p[lane], v + i);
    }
    if (v == 0xdeadbeef) p[64] = v;
}

template <int N_ACTIVE>
__launch_bounds__(256, 8) __global__ void add64_n(unsigned long long *p, int N_iters) {
    int lane = threadIdx.x & 31;
    if (lane >= N_ACTIVE) return;
    for (int i = 0; i < N_iters; i++) atomicAdd(&p[lane], 1ULL);
}

int main() {
    cudaSetDevice(0);
    int N_iters = 10000;
    unsigned long long *d_p; cudaMalloc(&d_p, 1024); cudaMemset(d_p, 0, 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;
    long total_warps = (long)blocks * threads / 32;

    auto run = [&](const char* name, void(*kfn)(unsigned long long*, int), int n_active, int distinct_addrs, int width) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_p, N_iters);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_p, N_iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long active_atomics = (long)blocks * threads * n_active * N_iters / 32;  // n_active per warp
        long warps = total_warps * N_iters;
        long pkts = warps * distinct_addrs;
        double T = active_atomics / (best/1000.0) / 1e9;
        double W = warps / (best/1000.0) / 1e9;
        double L = pkts / (best/1000.0) / 1e9;
        double L_per_video_cy = L / 1.860;
        double per_warp_ns = 1e9 / W;
        double per_warp_cy = per_warp_ns * 1.860;
        double payload_gbs = active_atomics * width / (best/1000.0) / 1e9;
        printf("  %-30s %.3f ms  T=%.2f Gops  W=%.2f Gwarp/s  L=%.2f Gpkt/s = %.2f L2pkt/cy  per-warp %.2f cy  payload %.0f GB/s\n",
            name, best, T, W, L, L_per_video_cy, per_warp_cy, payload_gbs);
    };

    printf("# Active-lane sweep: N lanes target N consecutive 8B addrs (= N*8 byte region)\n");
    printf("# All warps target SAME N*8 region (max contention on it)\n\n");
    printf("ADD64:\n");
    run("add64 1 lane (8B = 1/4 sec)",   add64_n<1>,   1,  1,  8);
    run("add64 2 lanes (16B = 1/2 sec)", add64_n<2>,   2,  2,  8);
    run("add64 4 lanes (32B = 1 sec)",   add64_n<4>,   4,  4,  8);
    run("add64 8 lanes (64B = 2 sec)",   add64_n<8>,   8,  8,  8);
    run("add64 16 lanes (128B = 1 line)", add64_n<16>, 16, 16, 8);
    run("add64 32 lanes (256B = 2 lines)", add64_n<32>, 32, 32, 8);
    printf("\nEXCH64:\n");
    run("exch64 1 lane",                 exch64_n<1>,  1,  1,  8);
    run("exch64 2 lanes",                exch64_n<2>,  2,  2,  8);
    run("exch64 4 lanes (32B = 1 sec)",  exch64_n<4>,  4,  4,  8);
    run("exch64 8 lanes (64B = 2 sec)",  exch64_n<8>,  8,  8,  8);
    run("exch64 16 lanes (128B = 1 line)", exch64_n<16>, 16, 16, 8);
    run("exch64 32 lanes (256B = 2 lines)", exch64_n<32>, 32, 32, 8);
    return 0;
}
