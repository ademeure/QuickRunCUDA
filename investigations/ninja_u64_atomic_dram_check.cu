#include <cuda_runtime.h>
#include <cstdio>
// Variant 1: ALL DISTINCT - 1 thread per cache line
__launch_bounds__(256, 8) __global__ void u64_distinct(unsigned long long *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 16) % N_addrs;  // 16 ull stride = 128B
        atomicAdd(&p[addr], 1ULL);
    }
}
// Variant 2: per-block 16 addrs, but spread blocks so each owns DIFFERENT line + iter cycles
__launch_bounds__(256, 8) __global__ void u64_perblock16_HBM(unsigned long long *p, int N_iters, long N_addrs, int N_lines) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 15;
    // Each block on its own line, but blocks cycle through MANY lines per iter
    // -> ensures lines don't fit in L2
    for (int i = 0; i < N_iters; i++) {
        long line = (blockIdx.x + (long)i * gridDim.x) % N_lines;
        long addr = (line * 16 + target_idx) % N_addrs;
        atomicAdd(&p[addr], 1ULL);
    }
}
// Variant 3: per-block-line BUT each iter same block→same line (L2 reuse — should NOT exceed)
__launch_bounds__(256, 8) __global__ void u64_perblock16_L2reuse(unsigned long long *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 15;
    long block_base = blockIdx.x * 16;
    long addr = (block_base + target_idx) % N_addrs;
    for (int i = 0; i < N_iters; i++) atomicAdd(&p[addr], 1ULL);
}
int main() {
    cudaSetDevice(0);
    int N_iters = 100;
    long N_addrs = 2L * 1024 * 1024 * 1024 / 8;  // 2 GB
    int N_lines_HBM = 2 * 1024 * 1024;  // 2M lines = 256 MB > L2 126MB
    unsigned long long *d_p; cudaMalloc(&d_p, (size_t)N_addrs * 8); cudaMemset(d_p, 0, (size_t)N_addrs * 8);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;
    auto run = [&](const char* name, auto kfn, int width) {
        for (int i = 0; i < 3; i++) kfn();
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        double T = ops / (best/1000.0) / 1e9;
        double payload = ops * width / (best/1000.0) / 1e9;
        printf("  %-40s %.3f ms  T=%.1f Gops  payload %.0f GB/s = %.2f TB/s\n",
            name, best, T, payload, payload/1000);
    };
    run("u64 ALL DISTINCT (lines spread)",
        [&](){ u64_distinct<<<blocks, threads>>>(d_p, N_iters, N_addrs); }, 8);
    run("u64 per-block-16 HBM-spread (L2 evict)",
        [&](){ u64_perblock16_HBM<<<blocks, threads>>>(d_p, N_iters, N_addrs, N_lines_HBM); }, 8);
    run("u64 per-block-16 L2-reuse (1 line/blk)",
        [&](){ u64_perblock16_L2reuse<<<blocks, threads>>>(d_p, N_iters, N_addrs); }, 8);
    return 0;
}
