// Real anti-DCE: dependency chain via low bits of read value
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void smem_read(int4 *out, int iters, int seed) {
    __shared__ int4 smem[256];
    int tid = threadIdx.x;
    smem[tid] = make_int4(tid + seed, tid + seed + 1, tid + seed + 2, tid + seed + 3);
    __syncthreads();

    int4 r0 = smem[tid];
    int4 r1 = smem[(tid+1) & 255];
    int4 r2 = smem[(tid+2) & 255];
    int4 r3 = smem[(tid+3) & 255];

    for (int i = 0; i < iters; i++) {
        // Address depends on previous read (true RAW dependency)
        // But masked to stay in 0..255 range
        int o = (r0.x ^ r1.x ^ r2.x ^ r3.x) & 0xff;  // depends on previous reads
        // We want a SINGLE address dependency — use 'o' to pick which warp-coalesced offset
        // Actually for SHMEM bandwidth, every thread should access different bank
        // tid + o + k where k varies per chain
        int4 n0 = smem[(tid + o + 0) & 255];
        int4 n1 = smem[(tid + o + 1) & 255];
        int4 n2 = smem[(tid + o + 2) & 255];
        int4 n3 = smem[(tid + o + 3) & 255];
        r0 = n0; r1 = n1; r2 = n2; r3 = n3;
    }
    if ((r0.x ^ r1.x ^ r2.x ^ r3.x) == 0xdeadbeef)
        out[blockIdx.x * blockDim.x + tid] = r0;
}

int main() {
    cudaSetDevice(0);
    int4 *d_out; cudaMalloc(&d_out, 148 * 256 * sizeof(int4));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 256;

    for (int i = 0; i < 3; i++) smem_read<<<blocks, threads>>>(d_out, iters, 1);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < 7; i++) {
        cudaEventRecord(e0);
        smem_read<<<blocks, threads>>>(d_out, iters, 1);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    // 4 reads per iter × 16 B = 64 B per thread per iter
    long total_bytes = (long)blocks * threads * iters * 4 * 16;
    double tb = total_bytes / (best/1000.0) / 1e12;
    printf("# SHMEM read with TRUE data-dep chain\n");
    printf("  time:        %.3f ms\n", best);
    printf("  total_bytes: %.2f GB\n", total_bytes/1e9);
    printf("  measured:    %.1f TB/s\n", tb);
    printf("  ratio:       %.1f%% of 38.5 TB/s theoretical\n", tb/38.5*100);
    return 0;
}
