// DEEP HBM atomic analysis: 32-bit vs 64-bit, varying combine ratio
// ALWAYS report Gops/s + payload bytes/s + DRAM bytes (separate ncu pass)
//
// HBM-resident: WS=1024 MB
// Combine ratio = N threads in warp targeting same cache line
//   stride per thread within warp = 32B / N (where 4B = 1 int slot)
#include <cuda_runtime.h>
#include <cstdio>

// Combine ratio = 32 (max): all threads in warp target consecutive ints (1 cache line)
__launch_bounds__(256, 8) __global__ void atom32_combine32(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_warps = ((long)gridDim.x * blockDim.x) / 32;
    int warp_id = tid / 32;
    int lane = tid & 31;
    for (int i = 0; i < N_iters; i++) {
        long base = (warp_id + (long)i * N_warps) * 32;
        atomicAdd(&p[(base + lane) % N_addrs], 1);
    }
}

// Combine ratio = 16: warp split 2 ways (lanes 0-15 → cache line A, lanes 16-31 → cache line B)
__launch_bounds__(256, 8) __global__ void atom32_combine16(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_warps = ((long)gridDim.x * blockDim.x) / 32;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int half = lane / 16;
    int sublane = lane & 15;
    for (int i = 0; i < N_iters; i++) {
        long base = ((warp_id * 2 + half) + (long)i * N_warps * 2) * 32;
        atomicAdd(&p[(base + sublane) % N_addrs], 1);
    }
}

// Combine ratio = 8
__launch_bounds__(256, 8) __global__ void atom32_combine8(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_warps = ((long)gridDim.x * blockDim.x) / 32;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int quarter = lane / 8;
    int sublane = lane & 7;
    for (int i = 0; i < N_iters; i++) {
        long base = ((warp_id * 4 + quarter) + (long)i * N_warps * 4) * 32;
        atomicAdd(&p[(base + sublane) % N_addrs], 1);
    }
}

// Combine = 1 (no combine, each thread its own cache line)
__launch_bounds__(256, 8) __global__ void atom32_combine1(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 32) % N_addrs;
        atomicAdd(&p[addr], 1);
    }
}

// 64-bit atomic, combine 16 (one cache line = 16 long longs)
__launch_bounds__(256, 8) __global__ void atom64_combine16(unsigned long long *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_warps = ((long)gridDim.x * blockDim.x) / 32;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int half = lane / 16;
    int sublane = lane & 15;
    for (int i = 0; i < N_iters; i++) {
        long base = ((warp_id * 2 + half) + (long)i * N_warps * 2) * 16;  // 16 ull per cache line
        atomicAdd(&p[(base + sublane) % N_addrs], 1ULL);
    }
}

// FP32 atomic combine32
__launch_bounds__(256, 8) __global__ void atomf_combine32(float *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_warps = ((long)gridDim.x * blockDim.x) / 32;
    int warp_id = tid / 32;
    int lane = tid & 31;
    for (int i = 0; i < N_iters; i++) {
        long base = (warp_id + (long)i * N_warps) * 32;
        atomicAdd(&p[(base + lane) % N_addrs], 1.0f);
    }
}

template <typename T>
double bench(const char* name, void(*kfn)(T*, int, long), T* d_p, long N_addrs, int width_bytes) {
    int blocks = 148 * 8, threads = 256;
    int N_iters = 100;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_p, N_iters, N_addrs);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s: %s\n", name, cudaGetErrorString(cudaGetLastError())); return 0; }
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
    double payload_gbs = ops * (double)width_bytes / (best/1000.0) / 1e9;
    printf("  %-30s  %.3f ms  %6.1f Gops  payload %5.0f GB/s (%dB ops)\n",
        name, best, gops, payload_gbs, width_bytes);
    return gops;
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    long WS_MB = (argc > 1) ? atol(argv[1]) : 1024;
    long N_int = WS_MB * 1024 * 1024 / 4;
    long N_ull = WS_MB * 1024 * 1024 / 8;
    int *d_pi; cudaMalloc(&d_pi, (size_t)N_int * 4);
    cudaMemset(d_pi, 0, (size_t)N_int * 4);
    unsigned long long *d_pu; cudaMalloc(&d_pu, (size_t)N_ull * 8);
    cudaMemset(d_pu, 0, (size_t)N_ull * 8);
    float *d_pf; cudaMalloc(&d_pf, (size_t)N_int * 4);
    cudaMemset(d_pf, 0, (size_t)N_int * 4);

    printf("# WS=%ldMB (HBM-resident) — atomic deep dive\n", WS_MB);
    bench<int>("int32  combine=32", atom32_combine32, d_pi, N_int, 4);
    bench<int>("int32  combine=16", atom32_combine16, d_pi, N_int, 4);
    bench<int>("int32  combine= 8", atom32_combine8,  d_pi, N_int, 4);
    bench<int>("int32  combine= 1", atom32_combine1,  d_pi, N_int, 4);
    bench<unsigned long long>("uint64 combine=16", atom64_combine16, d_pu, N_ull, 8);
    bench<float>("fp32   combine=32", atomf_combine32, d_pf, N_int, 4);
    return 0;
}
