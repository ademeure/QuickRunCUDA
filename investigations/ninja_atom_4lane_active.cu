// Only threads 0-3 active per warp (others early-exit)
// Each of t0/t1/t2/t3 targets unique 8-byte address (= 32B sector total)
#include <cuda_runtime.h>
#include <cstdio>

// 4 active lanes (t0123), each → unique uint64 addr
__launch_bounds__(256, 8) __global__ void exch64_4lanes(unsigned long long *p, int N_iters) {
    int lane = threadIdx.x & 31;
    if (lane >= 4) return;  // ONLY t0,t1,t2,t3 active
    int target_idx = lane;  // each → addrs 0,1,2,3
    unsigned long long v = (unsigned long long)threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        v = atomicExch(&p[target_idx], v + i);
    }
    if (v == 0xdeadbeef) p[16] = v;
}

// Same but ADD instead of EXCH
__launch_bounds__(256, 8) __global__ void add64_4lanes(unsigned long long *p, int N_iters) {
    int lane = threadIdx.x & 31;
    if (lane >= 4) return;
    int target_idx = lane;
    for (int i = 0; i < N_iters; i++) {
        atomicAdd(&p[target_idx], 1ULL);
    }
}

// 1 active lane (t0 only) for baseline
__launch_bounds__(256, 8) __global__ void exch64_1lane(unsigned long long *p, int N_iters) {
    int lane = threadIdx.x & 31;
    if (lane >= 1) return;
    unsigned long long v = (unsigned long long)threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        v = atomicExch(&p[0], v + i);
    }
    if (v == 0xdeadbeef) p[16] = v;
}

__launch_bounds__(256, 8) __global__ void add64_1lane(unsigned long long *p, int N_iters) {
    int lane = threadIdx.x & 31;
    if (lane >= 1) return;
    for (int i = 0; i < N_iters; i++) atomicAdd(&p[0], 1ULL);
}

int main() {
    cudaSetDevice(0);
    int N_iters = 10000;
    unsigned long long *d_p; cudaMalloc(&d_p, 1024); cudaMemset(d_p, 0, 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run = [&](const char* name, void(*kfn)(unsigned long long*, int), int active_lanes_per_warp, int distinct_addrs) {
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
        // Active thread atomics (only count active lanes)
        long active_threads = (long)blocks * threads * active_lanes_per_warp / 32;
        long ops = active_threads * N_iters;
        long warps = (long)blocks * threads / 32 * N_iters;  // total warps
        // L2 packets: distinct addrs per warp (not popc-merged because diff addrs)
        long pkts = warps * distinct_addrs;
        double T = ops / (best/1000.0) / 1e9;
        double W = warps / (best/1000.0) / 1e9;
        double L = pkts / (best/1000.0) / 1e9;
        double L_per_video_cy = L / 1.860;
        double payload_gbs = ops * 8.0 / (best/1000.0) / 1e9;
        printf("  %-30s %.3f ms  T=%.2f Gops  W=%.2f Gwarp/s  L=%.2f Gpkt/s = %.2f L2pkt/cy  payload %.0f GB/s\n",
            name, best, T, W, L, L_per_video_cy, payload_gbs);
    };

    printf("# 4-active-lane vs 1-active-lane atomic (others early-exit)\n");
    printf("# 256 threads/block but only N lanes actually issue\n\n");
    run("add64 1 lane (1 addr)",      add64_1lane,  1, 1);
    run("exch64 1 lane (1 addr)",     exch64_1lane, 1, 1);
    run("add64 4 lanes (4 distinct)", add64_4lanes, 4, 4);
    run("exch64 4 lanes (4 distinct)", exch64_4lanes, 4, 4);
    return 0;
}
