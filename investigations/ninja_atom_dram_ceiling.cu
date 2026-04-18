// Push DRAM atomic BW past 5.46 TB/s
#include <cuda_runtime.h>
#include <cstdio>

// 64-bit, NO combining (each thread → different cache line)
__launch_bounds__(256, 8) __global__ void k64_uncomb(unsigned long long *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 16) % N_addrs;  // 16 ull stride = 128B = cache line
        atomicAdd(&p[addr], 1ULL);
    }
}

// 32-bit, NO combining (baseline)
__launch_bounds__(256, 8) __global__ void k32_uncomb(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 32) % N_addrs;
        atomicAdd(&p[addr], 1);
    }
}

// 32-bit, combine=2 (warp split into 16 pairs, 16 cache lines per warp)
__launch_bounds__(256, 8) __global__ void k32_c2(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_warps = ((long)gridDim.x * blockDim.x) / 32;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int half = lane / 2;  // 16 cache lines (one per pair of lanes)
    int sublane = lane & 1;
    for (int i = 0; i < N_iters; i++) {
        long base = ((warp_id * 16 + half) + (long)i * N_warps * 16) * 32;
        atomicAdd(&p[(base + sublane) % N_addrs], 1);
    }
}

template <typename T>
double bench(const char* name, void(*kfn)(T*, int, long), T* d_p, long N_addrs, int width_bytes, int amp_bytes_per_op) {
    int blocks = 148 * 8, threads = 256, N_iters = 100;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_p, N_iters, N_addrs);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return 0; }
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        kfn<<<blocks, threads>>>(d_p, N_iters, N_addrs);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = (long)blocks * threads * N_iters;
    double gops = ops / (best/1000.0) / 1e9;
    double payload_gbs = ops * width_bytes / (best/1000.0) / 1e9;
    double pred_dram_gbs = ops * amp_bytes_per_op / (best/1000.0) / 1e9;
    printf("  %-22s %.3f ms  %.1f Gops/s  payload %.0f GB/s  predicted DRAM %.0f GB/s\n",
        name, best, gops, payload_gbs, pred_dram_gbs);
    return gops;
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    long WS_MB = (argc > 1) ? atol(argv[1]) : 2048;
    long N_int = WS_MB * 1024 * 1024 / 4;
    long N_ull = WS_MB * 1024 * 1024 / 8;
    int *d_p32; cudaMalloc(&d_p32, (size_t)N_int * 4); cudaMemset(d_p32, 0, (size_t)N_int * 4);
    unsigned long long *d_p64; cudaMalloc(&d_p64, (size_t)N_ull * 8); cudaMemset(d_p64, 0, (size_t)N_ull * 8);
    printf("# WS=%ldMB HBM-resident\n", WS_MB);
    bench<int>("int32 combine=1", k32_uncomb, d_p32, N_int, 4, 256);  // each = full RMW line
    bench<int>("int32 combine=2", k32_c2, d_p32, N_int, 4, 256);
    bench<unsigned long long>("uint64 combine=1", k64_uncomb, d_p64, N_ull, 8, 256);
    return 0;
}
