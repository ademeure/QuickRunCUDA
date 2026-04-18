// Combine: many distinct lines (use all L2 units) + many distinct addrs per line (use all 4 slots)
#include <cuda_runtime.h>
#include <cstdio>

// Each block targets its OWN cache line (different line per block)
// Within block: 4 distinct addrs in 32B sector (use all 4 slots of that sector)
// All warps in block contend on those 4 addrs -> fully use 1 sector unit
__launch_bounds__(256, 8) __global__ void chip_max(unsigned long long *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx_in_sector = lane & 3;  // 4 distinct ull addresses
    long block_base = blockIdx.x * 4;     // each block on its own 32B sector
    long addr = (block_base + target_idx_in_sector) % N_addrs;
    for (int i = 0; i < N_iters; i++) {
        atomicAdd(&p[addr], 1ULL);
    }
}

// All blocks targeting all-distinct lines spread across L2
// Within block: 16 distinct addrs (1 line, 4 slots used heavily)
__launch_bounds__(256, 8) __global__ void chip_max16(unsigned long long *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 15;  // 16 distinct ull = 1 cache line
    long block_base = blockIdx.x * 16;
    long addr = (block_base + target_idx) % N_addrs;
    for (int i = 0; i < N_iters; i++) {
        atomicAdd(&p[addr], 1ULL);
    }
}

// Each thread = unique cache line spread (this is the prior multi-line baseline)
__launch_bounds__(256, 8) __global__ void all_distinct(unsigned long long *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 16) % N_addrs;
        atomicAdd(&p[addr], 1ULL);
    }
}

int main() {
    cudaSetDevice(0);
    long WS_MB = 1024;
    long N = WS_MB * 1024 * 1024 / 8;
    unsigned long long *d_p; cudaMalloc(&d_p, (size_t)N * 8); cudaMemset(d_p, 0, (size_t)N * 8);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256, N_iters = 100;

    auto run = [&](const char* name, void(*kfn)(unsigned long long*, int, long)) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_p, N_iters, N);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_p, N_iters, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        double T = ops / (best/1000.0) / 1e9;
        double T_per_video_cy = T / 1.860;
        double payload = ops * 8.0 / (best/1000.0) / 1e9;
        printf("  %-40s %.3f ms  T=%.1f Gops  T/video-cy=%.1f  payload %.0f GB/s\n",
            name, best, T, T_per_video_cy, payload);
    };

    printf("# Push for L2 atomic chip-max (HBM-resident WS=1024MB)\n");
    printf("# Each block on own line + within-block contention to fill all 4 slots\n\n");
    run("ALL DISTINCT (1 thread/line)",          all_distinct);
    run("Per-block 4 addrs/sector (sector use)", chip_max);
    run("Per-block 16 addrs/line (line use)",    chip_max16);
    return 0;
}
