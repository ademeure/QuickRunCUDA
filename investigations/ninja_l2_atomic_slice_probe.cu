// Measure L2 atomic unit count by forcing single-slice vs all-slice contention
//
// Per catalog C2: L2 hash interleave at 4 KiB granularity
// Strategy:
//   Test SAME-SLICE: all atomics go to SAME 4 KB region → 1 L2 slice
//     → throughput = per-slice peak
//   Test ALL-SLICES: atomics spread across all of L2 → all slices in parallel
//     → throughput = per-slice * num_slices
//   Ratio = num_slices = answer
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// ALL-SLICES: each thread → unique 4 KB page (so different L2 slice)
__launch_bounds__(256, 8) __global__ void atom_all_slices(int *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 1024) % N_addrs;  // 1024 ints * 4B = 4096 B (4 KB stride)
        atomicAdd(&p[addr], 1);
    }
}

// ONE-SLICE: all atomics in first 4 KB region (1024 distinct ints, all same L2 slice)
__launch_bounds__(256, 8) __global__ void atom_one_slice(int *p, int N_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        // Each thread cycles through 1024 ints in the first 4 KB page
        // Different threads target different ints → no thread-level conflict
        // But ALL hit same L2 slice (same 4 KB)
        int addr = (tid + i * 31) & 1023;  // mod 1024
        atomicAdd(&p[addr], 1);
    }
}

// FEW-SLICES (control): use first 8 KB → 2 distinct L2 slices
__launch_bounds__(256, 8) __global__ void atom_two_slices(int *p, int N_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        int addr = (tid + i * 31) & 2047;  // 2 × 4 KB
        atomicAdd(&p[addr], 1);
    }
}

// 4-slices, 8-slices, 16-slices, ...
template <int N_PAGES>
__launch_bounds__(256, 8) __global__ void atom_n_slices(int *p, int N_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int mask = N_PAGES * 1024 - 1;
    for (int i = 0; i < N_iters; i++) {
        int addr = (tid + i * 31) & mask;
        atomicAdd(&p[addr], 1);
    }
}

int main() {
    cudaSetDevice(0);
    long N = 1024L * 1024 * 1024 / 4;  // 1 GB int array
    int *d_p; cudaMalloc(&d_p, (size_t)N * 4);
    cudaMemset(d_p, 0, (size_t)N * 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256, N_iters = 100;

    auto run = [&](const char* name, void(*kfn)(int*, int)) {
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
        double gops = ops / (best/1000.0) / 1e9;
        printf("  %-30s %.4f ms  %.1f Gops/s thread-atomics\n", name, best, gops);
    };

    auto run_all = [&](const char* name, void(*kfn)(int*, int, long)) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_p, N_iters, N);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) return;
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_p, N_iters, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        double gops = ops / (best/1000.0) / 1e9;
        printf("  %-30s %.4f ms  %.1f Gops/s thread-atomics\n", name, best, gops);
    };

    printf("# L2 atomic unit count probe (1024 MB WS array)\n");
    run("ONE 4KB slice (1 L2 unit?)",   atom_one_slice);
    run("TWO 4KB slices (2 L2 units?)", atom_two_slices);
    run("4 slices",   atom_n_slices<4>);
    run("8 slices",   atom_n_slices<8>);
    run("16 slices",  atom_n_slices<16>);
    run("32 slices",  atom_n_slices<32>);
    run("64 slices",  atom_n_slices<64>);
    run("128 slices", atom_n_slices<128>);
    run("256 slices", atom_n_slices<256>);
    run_all("ALL slices (1 GB spread)", atom_all_slices);
    return 0;
}
