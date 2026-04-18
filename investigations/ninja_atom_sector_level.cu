// Atomic where all warps target 4 consecutive 64-bit addresses (= 32B sector)
// Pattern: lane % 4 picks one of 4 distinct ull addresses
// Within warp: 8 lanes per address (popc-merge to 1 atomic per address per warp)
// All 4 addresses in same 32B sector
#include <cuda_runtime.h>
#include <cstdio>

// 64-bit (uint64) at 4 consecutive addresses (= 32B = 1 sector)
__launch_bounds__(256, 8) __global__ void atom64_4addr(unsigned long long *p, int N_iters) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 3;  // 4 distinct ull addresses (0,1,2,3) = 32B
    for (int i = 0; i < N_iters; i++) {
        atomicAdd(&p[target_idx], 1ULL);
    }
}

// 32-bit (int) at 8 consecutive addresses (= 32B = 1 sector)
__launch_bounds__(256, 8) __global__ void atom32_8addr(int *p, int N_iters) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 7;  // 8 distinct int addresses = 32B sector
    for (int i = 0; i < N_iters; i++) {
        atomicAdd(&p[target_idx], 1);
    }
}

// 32-bit, all SAME address (popc-merge to 1 op per warp) — baseline
__launch_bounds__(256, 8) __global__ void atom32_1addr(int *p, int N_iters) {
    for (int i = 0; i < N_iters; i++) atomicAdd(&p[0], 1);
}

// 32-bit, full warp distinct (no merge)
__launch_bounds__(256, 8) __global__ void atom32_32addr(int *p, int N_iters) {
    int lane = threadIdx.x & 31;  // 32 distinct ints = 128B = full cache line
    for (int i = 0; i < N_iters; i++) {
        atomicAdd(&p[lane], 1);
    }
}

int main() {
    cudaSetDevice(0);
    int N_iters = 1000;
    int *d_p32; cudaMalloc(&d_p32, 1024); cudaMemset(d_p32, 0, 1024);
    unsigned long long *d_p64; cudaMalloc(&d_p64, 1024); cudaMemset(d_p64, 0, 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run = [&](const char* name, void(*kfn32)(int*, int), void(*kfn64)(unsigned long long*, int), int width_bytes, int distinct_addrs_per_warp) {
        for (int i = 0; i < 3; i++) {
            if (kfn32) kfn32<<<blocks, threads>>>(d_p32, N_iters);
            else kfn64<<<blocks, threads>>>(d_p64, N_iters);
        }
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            if (kfn32) kfn32<<<blocks, threads>>>(d_p32, N_iters);
            else kfn64<<<blocks, threads>>>(d_p64, N_iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        long warps = (long)blocks * threads / 32 * N_iters;
        double T = ops / (best/1000.0) / 1e9;
        double W = warps / (best/1000.0) / 1e9;
        // L2 packets: per warp = distinct addresses per warp (popc-merged)
        double L = warps * distinct_addrs_per_warp / (best/1000.0) / 1e9;
        double L_per_video_cy = L / 1.860;
        double payload_gbs = ops * width_bytes / (best/1000.0) / 1e9;
        printf("  %-30s %.3f ms  T=%.1f Gops  W=%.2f Gwarp/s  L=%.2f Gpkt/s = %.2f L2pkt/cy  payload %.0f GB/s\n",
            name, best, T, W, L, L_per_video_cy, payload_gbs);
    };

    printf("# Atomic locality sweep — all warps to single line, varying distinct addresses\n");
    printf("# Width: 32B sector = 8 ints = 4 ulls = 1/4 of 128B cache line\n\n");
    run("int32 1 addr (full merge)",  atom32_1addr,  nullptr, 4, 1);
    run("int32 8 addr (32B sector)",  atom32_8addr,  nullptr, 4, 8);
    run("int32 32 addr (full line)",  atom32_32addr, nullptr, 4, 32);
    run("uint64 4 addr (32B sector)", nullptr, atom64_4addr, 8, 4);
    return 0;
}
