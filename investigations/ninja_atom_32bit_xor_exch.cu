// 32-bit int variants for exch and xor
#include <cuda_runtime.h>
#include <cstdio>

// 32-bit XOR
__launch_bounds__(256, 8) __global__ void xor32_all_distinct(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 32) % N_addrs;  // 32 ints = 128B
        atomicXor(&p[addr], 1);
    }
}

__launch_bounds__(256, 8) __global__ void xor32_block_8addr(int *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 7;  // 8 ints = 32B sector
    long block_base = blockIdx.x * 8;
    long addr = (block_base + target_idx) % N_addrs;
    for (int i = 0; i < N_iters; i++) atomicXor(&p[addr], 1);
}

__launch_bounds__(256, 8) __global__ void xor32_block_32addr(int *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 31;  // 32 ints = 128B = 1 line
    long block_base = blockIdx.x * 32;
    long addr = (block_base + target_idx) % N_addrs;
    for (int i = 0; i < N_iters; i++) atomicXor(&p[addr], 1);
}

// 32-bit EXCH
__launch_bounds__(256, 8) __global__ void exch32_all_distinct(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    int v = (int)tid;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 32) % N_addrs;
        v = atomicExch(&p[addr], v + i);
    }
    if (v == 0xdeadbeef) p[0] = v;
}

__launch_bounds__(256, 8) __global__ void exch32_block_8addr(int *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 7;
    long block_base = blockIdx.x * 8;
    long addr = (block_base + target_idx) % N_addrs;
    int v = lane;
    for (int i = 0; i < N_iters; i++) {
        v = atomicExch(&p[addr], v + i);
    }
    if (v == 0xdeadbeef) p[0] = v;
}

__launch_bounds__(256, 8) __global__ void exch32_block_32addr(int *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 31;
    long block_base = blockIdx.x * 32;
    long addr = (block_base + target_idx) % N_addrs;
    int v = lane;
    for (int i = 0; i < N_iters; i++) {
        v = atomicExch(&p[addr], v + i);
    }
    if (v == 0xdeadbeef) p[0] = v;
}

int main() {
    cudaSetDevice(0);
    int N_iters = 100;
    long N = 1024L * 1024 * 1024 / 4;
    int *d_p; cudaMalloc(&d_p, (size_t)N * 4); cudaMemset(d_p, 0, (size_t)N * 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run = [&](const char* name, void(*kfn)(int*, int, long)) {
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
        double payload = ops * 4.0 / (best/1000.0) / 1e9;  // 4B int
        printf("  %-35s %.3f ms  T=%.1f Gops  T/video-cy=%.1f  payload %.0f GB/s\n",
            name, best, T, T_per_video_cy, payload);
    };

    printf("# 32-bit XOR variants:\n");
    run("xor32 ALL DISTINCT (1 thread/line)", xor32_all_distinct);
    run("xor32 per-block 8 addrs/sector",     xor32_block_8addr);
    run("xor32 per-block 32 addrs/line",      xor32_block_32addr);

    printf("\n# 32-bit EXCH variants:\n");
    run("exch32 ALL DISTINCT",                exch32_all_distinct);
    run("exch32 per-block 8 addrs/sector",    exch32_block_8addr);
    run("exch32 per-block 32 addrs/line",     exch32_block_32addr);

    return 0;
}
