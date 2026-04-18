// uint64 atomic EXCHANGE with 4 addresses (t0123 = t4567 = ... etc)
// 32 lanes / 4 addrs = 8 lanes per address
// EXCH cannot popc-merge → each lane is its own L2 packet
#include <cuda_runtime.h>
#include <cstdio>

// 64-bit atomicExch (no merge possible)
__launch_bounds__(256, 8) __global__ void exch64_4addr(unsigned long long *p, int N_iters) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 3;  // 4 distinct ull = 32B sector
    unsigned long long v = (unsigned long long)threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        v = atomicExch(&p[target_idx], v + i);
    }
    if (v == 0xdeadbeef) p[16] = v;
}

// Compare: 64-bit atomicAdd to same 4 addrs (popc-merge IS possible)
__launch_bounds__(256, 8) __global__ void add64_4addr(unsigned long long *p, int N_iters) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 3;
    for (int i = 0; i < N_iters; i++) {
        atomicAdd(&p[target_idx], 1ULL);
    }
}

// Single address baselines
__launch_bounds__(256, 8) __global__ void exch64_1addr(unsigned long long *p, int N_iters) {
    unsigned long long v = (unsigned long long)threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        v = atomicExch(&p[0], v + i);
    }
    if (v == 0xdeadbeef) p[16] = v;
}

__launch_bounds__(256, 8) __global__ void add64_1addr(unsigned long long *p, int N_iters) {
    for (int i = 0; i < N_iters; i++) atomicAdd(&p[0], 1ULL);
}

int main() {
    cudaSetDevice(0);
    int N_iters = 1000;
    unsigned long long *d_p; cudaMalloc(&d_p, 1024); cudaMemset(d_p, 0, 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run = [&](const char* name, void(*kfn)(unsigned long long*, int), int distinct_per_warp_pre_merge, int distinct_per_warp_post_merge) {
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
        long ops = (long)blocks * threads * N_iters;
        long warps = (long)blocks * threads / 32 * N_iters;
        double T = ops / (best/1000.0) / 1e9;
        double W = warps / (best/1000.0) / 1e9;
        double L = warps * distinct_per_warp_post_merge / (best/1000.0) / 1e9;
        double L_per_video_cy = L / 1.860;
        double payload_gbs = ops * 8.0 / (best/1000.0) / 1e9;
        printf("  %-30s %.3f ms  T=%.1f Gops  W=%.2f Gwarp/s  L=%.2f Gpkt/s = %.2f L2pkt/cy  payload %.0f GB/s\n",
            name, best, T, W, L, L_per_video_cy, payload_gbs);
    };

    printf("# 64-bit atomic single-line: ADD vs EXCH at 1, 4 addresses\n");
    printf("# (32 lanes / 4 addrs = 8 lanes per addr; ADD popc-merges 8 lanes -> 1, EXCH does not)\n\n");
    run("add64 1 addr (full merge)",  add64_1addr,  32, 1);
    run("exch64 1 addr (no merge)",   exch64_1addr, 32, 32);
    run("add64 4 addr (8 merge per)", add64_4addr,  32, 4);
    run("exch64 4 addr (no merge, 32 pkts)", exch64_4addr, 32, 32);

    return 0;
}
