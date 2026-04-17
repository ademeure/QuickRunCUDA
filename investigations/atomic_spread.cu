// Test atomic throughput with deliberate cache-line spacing
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void atom_spread(unsigned *target, int iters, int stride) {
    int wid = blockIdx.x * (blockDim.x/32) + threadIdx.x/32;
    unsigned *my_target = target + wid * stride;
    for (int i = 0; i < iters; i++) {
        atomicAdd(my_target, 1);
    }
}

int main() {
    cudaSetDevice(0);

    int n_warps = 148 * 4;  // 4 warps per block
    int max_target_size = n_warps * 1024;  // worst case stride
    unsigned *d_target;
    cudaMalloc(&d_target, max_target_size * sizeof(unsigned));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int iters = 1000;
    int blocks = 148, threads = 128;
    long total_atoms = (long)blocks * threads * iters;

    auto bench = [&](int stride) {
        cudaMemset(d_target, 0, max_target_size * sizeof(unsigned));
        for (int i = 0; i < 3; i++) atom_spread<<<blocks, threads>>>(d_target, iters, stride);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            cudaMemset(d_target, 0, max_target_size * sizeof(unsigned));
            cudaEventRecord(e0);
            atom_spread<<<blocks, threads>>>(d_target, iters, stride);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double gops = total_atoms / (best/1000.0) / 1e9;
        return gops;
    };

    printf("# B300 per-warp atomic throughput vs target stride (cache line spacing)\n");
    printf("# stride 1 = consecutive (32 of 4B in same cache line)\n");
    printf("# stride 32 = one cache line apart (1 atomic per line)\n\n");
    printf("# %-15s %-15s %-15s\n", "stride_words", "stride_bytes", "Gatomic/s");

    for (int s : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
        double gops = bench(s);
        printf("  %-15d %-15d %-15.2f\n", s, s*4, gops);
    }

    return 0;
}
