// Peak SHMEM bandwidth measurement
#include <cuda_runtime.h>
#include <cstdio>

__global__ void smem_peak(int *out, int iters, int seed) {
    __shared__ int4 smem[256];
    int tid = threadIdx.x;

    // Initialize
    smem[tid] = make_int4(tid + seed, tid + seed + 1, tid + seed + 2, tid + seed + 3);
    __syncthreads();

    int4 a = smem[tid];
    int4 b = smem[(tid + 1) & 255];
    int4 c = smem[(tid + 2) & 255];
    int4 d = smem[(tid + 3) & 255];

    for (int i = 0; i < iters; i++) {
        // Mix: read 4 int4s (16 B each = 64 B total per iter per thread)
        int idx = (tid + i) & 255;
        int4 r0 = smem[idx];
        int4 r1 = smem[(idx + 8) & 255];
        int4 r2 = smem[(idx + 16) & 255];
        int4 r3 = smem[(idx + 24) & 255];

        a.x ^= r0.x; a.y ^= r0.y; a.z ^= r0.z; a.w ^= r0.w;
        b.x ^= r1.x; b.y ^= r1.y; b.z ^= r1.z; b.w ^= r1.w;
        c.x ^= r2.x; c.y ^= r2.y; c.z ^= r2.z; c.w ^= r2.w;
        d.x ^= r3.x; d.y ^= r3.y; d.z ^= r3.z; d.w ^= r3.w;

        // Write back to defeat DCE
        smem[(idx + tid) & 255].x = a.x ^ b.x ^ c.x ^ d.x;
    }

    if (a.x ^ a.y ^ a.z ^ a.w == 0xdeadbeef)
        out[blockIdx.x * blockDim.x + tid] = a.x ^ b.x ^ c.x ^ d.x;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 148 * 256 * sizeof(int));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 256;

    for (int i = 0; i < 3; i++) smem_peak<<<blocks, threads>>>(d_out, iters, 1);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        smem_peak<<<blocks, threads>>>(d_out, iters, 1);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    // Per iter: 4 int4 reads = 64 B, plus 1 int4 write = 16 B (well, 4B for partial), so ~80 B/iter
    // Actually the write is to a single int (4 B) — total ~68 B
    long total_bytes = (long)blocks * threads * iters * 68;
    double tb = total_bytes / (best/1000.0) / 1e12;

    printf("# B300 SHMEM peak bandwidth\n");
    printf("# 148 blocks × 256 threads × 100k iter × (4 int4 reads + 1 int write) = 68 B/iter\n\n");
    printf("  Time: %.3f ms\n", best);
    printf("  Aggregate BW: %.1f TB/s (theoretical 38.5 TB/s)\n", tb);

    return 0;
}
