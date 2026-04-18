// Chip-max pattern (per-block own line) but L2-resident
// Many unique addresses per thread/warp, all warm
#include <cuda_runtime.h>
#include <cstdio>

// u32 chip-max per-block 32 addrs/line
__launch_bounds__(256, 8) __global__ void u32_chipmax(int *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 31;  // 32 distinct ints / line
    long block_base = blockIdx.x * 32;
    long addr = (block_base + target_idx) % N_addrs;
    for (int i = 0; i < N_iters; i++) atomicAdd(&p[addr], 1);
}

// u64 chip-max per-block 16 addrs/line
__launch_bounds__(256, 8) __global__ void u64_chipmax(unsigned long long *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 15;
    long block_base = blockIdx.x * 16;
    long addr = (block_base + target_idx) % N_addrs;
    for (int i = 0; i < N_iters; i++) atomicAdd(&p[addr], 1ULL);
}

// Multi-address warm: each warp's lanes target different L2 lines (1 line per lane), all in WS
__launch_bounds__(256, 8) __global__ void u32_multiline_warm(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 32) % N_addrs;  // different cache line per thread
        atomicAdd(&p[addr], 1);
    }
}

__launch_bounds__(256, 8) __global__ void u64_multiline_warm(unsigned long long *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 16) % N_addrs;
        atomicAdd(&p[addr], 1ULL);
    }
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    long WS_MB = (argc > 1) ? atol(argv[1]) : 64;
    long N_int = WS_MB * 1024 * 1024 / 4;
    long N_ull = WS_MB * 1024 * 1024 / 8;
    int *d_p32; cudaMalloc(&d_p32, (size_t)N_int * 4); cudaMemset(d_p32, 0, (size_t)N_int * 4);
    unsigned long long *d_p64; cudaMalloc(&d_p64, (size_t)N_ull * 8); cudaMemset(d_p64, 0, (size_t)N_ull * 8);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256, N_iters = 100;

    auto run = [&](const char* name, auto kfn, int width) {
        for (int i = 0; i < 5; i++) kfn();  // warm-up
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR\n"); return; }
        float best = 1e30f;
        for (int i = 0; i < 8; i++) {
            cudaEventRecord(e0);
            kfn();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        double T = ops / (best/1000.0) / 1e9;
        double payload = ops * width / (best/1000.0) / 1e9;
        printf("  %-25s WS=%4ldMB  %.3f ms  T=%.0f Gops  payload %.0f GB/s = %.2f TB/s\n",
            name, WS_MB, best, T, payload, payload/1000);
    };

    printf("# WARM-L2 chip-max + multi-line patterns (L2 size 126 MiB)\n\n");
    run("u32 chipmax (per-block-32)",
        [&]{u32_chipmax<<<blocks, threads>>>(d_p32, N_iters, N_int);}, 4);
    run("u32 multiline (1 thread/line)",
        [&]{u32_multiline_warm<<<blocks, threads>>>(d_p32, N_iters, N_int);}, 4);
    run("u64 chipmax (per-block-16)",
        [&]{u64_chipmax<<<blocks, threads>>>(d_p64, N_iters, N_ull);}, 8);
    run("u64 multiline (1 thread/line)",
        [&]{u64_multiline_warm<<<blocks, threads>>>(d_p64, N_iters, N_ull);}, 8);
    return 0;
}
