// Sweep R:W ratio — does any ratio escape 6.8 TB/s aggregate?
#include <cuda_runtime.h>
#include <cstdio>

// Each thread does N_R reads + N_W writes (each = 32 B v8)
// Per-warp coalesced; same data layout as prior rigor tests
template<int N_R, int N_W>
__launch_bounds__(256, 8) __global__ void rw_ratio(int *src, int *dst, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *sb = src + warp_id * (N_R * 32 * 8);  // unique read region
    int *db = dst + warp_id * (N_W * 32 * 8);  // unique write region
    int acc = 0;

    // Reads first
    int loaded[N_R > 0 ? N_R : 1][8];
    #pragma unroll
    for (int it = 0; it < N_R; it++) {
        int *p = sb + (it * 32 + lane) * 8;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(loaded[it][0]),"=r"(loaded[it][1]),"=r"(loaded[it][2]),"=r"(loaded[it][3]),
              "=r"(loaded[it][4]),"=r"(loaded[it][5]),"=r"(loaded[it][6]),"=r"(loaded[it][7])
            : "l"(p));
        acc ^= loaded[it][0];
    }
    // Then writes (data depends on reads to prevent DCE)
    int v = acc + 1;
    #pragma unroll
    for (int it = 0; it < N_W; it++) {
        int *p = db + (it * 32 + lane) * 8;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
}

template<int N_R, int N_W>
void run(int blocks, int threads, int *d_a, int *d_b, int *d_out, size_t total_threads) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) rw_ratio<N_R, N_W><<<blocks, threads>>>(d_a, d_b, d_out);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        rw_ratio<N_R, N_W><<<blocks, threads>>>(d_a, d_b, d_out);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    // Bytes touched per thread = (N_R + N_W) * 32
    long bytes_r = (long)total_threads * N_R * 32;
    long bytes_w = (long)total_threads * N_W * 32;
    long bytes_total = bytes_r + bytes_w;
    double r_bw = bytes_r / (best/1000) / 1e9;
    double w_bw = bytes_w / (best/1000) / 1e9;
    double agg = bytes_total / (best/1000) / 1e9;
    printf("  R=%2d W=%2d: %.3f ms  %5.0f R + %5.0f W = %5.0f GB/s aggregate\n",
           N_R, N_W, best, r_bw, w_bw, agg);
}

int main() {
    cudaSetDevice(0);
    size_t bytes_per_buf = 4096ul * 1024 * 1024;
    int *d_a; cudaMalloc(&d_a, bytes_per_buf); cudaMemset(d_a, 0xab, bytes_per_buf);
    int *d_b; cudaMalloc(&d_b, bytes_per_buf);
    int *d_out; cudaMalloc(&d_out, 1024 * 1024 * sizeof(int));

    // Each thread: per-warp 32-lane coalesced reads/writes of 32B each
    // Block has 256 thr = 8 warps; each warp owns N_R*32*8 ints for reads, same for writes
    int threads = 256;
    // Choose blocks so total threads × max(N_R, N_W) × 32 B <= bytes_per_buf
    // For N=32: warps × 32*32*8 = warps * 8192 ints = warps * 32 KB. Total = 4 GB / 32 KB = 131072 warps = 16384 blocks
    int blocks = 16384;
    size_t total_threads = (size_t)blocks * threads;

    printf("# Sweep R:W ratio, total mem ops constant at 32 per thread\n");
    printf("# 16384 blocks × 256 thr = %ld warps; 32 B per op per lane\n\n", total_threads/32);

    run<32, 0>(blocks, threads, d_a, d_b, d_out, total_threads);
    run<28, 4>(blocks, threads, d_a, d_b, d_out, total_threads);
    run<24, 8>(blocks, threads, d_a, d_b, d_out, total_threads);
    run<20, 12>(blocks, threads, d_a, d_b, d_out, total_threads);
    run<16, 16>(blocks, threads, d_a, d_b, d_out, total_threads);
    run<12, 20>(blocks, threads, d_a, d_b, d_out, total_threads);
    run<8, 24>(blocks, threads, d_a, d_b, d_out, total_threads);
    run<4, 28>(blocks, threads, d_a, d_b, d_out, total_threads);
    run<0, 32>(blocks, threads, d_a, d_b, d_out, total_threads);

    return 0;
}
