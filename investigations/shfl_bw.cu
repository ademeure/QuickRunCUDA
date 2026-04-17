// Warp shuffle throughput on B300
#include <cuda_runtime.h>
#include <cstdio>

__global__ void shfl_xor_chain(unsigned *out, int iters, unsigned mask) {
    unsigned a = threadIdx.x + 1;
    unsigned b = threadIdx.x * 7 + 1;
    unsigned c = threadIdx.x * 13 + 1;
    unsigned d = threadIdx.x * 31 + 1;
    for (int i = 0; i < iters; i++) {
        a = __shfl_xor_sync(0xffffffff, a, mask);
        b = __shfl_xor_sync(0xffffffff, b, mask);
        c = __shfl_xor_sync(0xffffffff, c, mask);
        d = __shfl_xor_sync(0xffffffff, d, mask);
    }
    if (a + b + c + d == 0xdeadbeef) out[blockIdx.x] = a+b+c+d;
}

__global__ void shfl_idx_chain(unsigned *out, int iters, int src) {
    unsigned a = threadIdx.x + 1;
    unsigned b = threadIdx.x * 7 + 1;
    unsigned c = threadIdx.x * 13 + 1;
    unsigned d = threadIdx.x * 31 + 1;
    for (int i = 0; i < iters; i++) {
        a = __shfl_sync(0xffffffff, a, src);
        b = __shfl_sync(0xffffffff, b, src);
        c = __shfl_sync(0xffffffff, c, src);
        d = __shfl_sync(0xffffffff, d, src);
    }
    if (a + b + c + d == 0xdeadbeef) out[blockIdx.x] = a+b+c+d;
}

__global__ void shfl_up_chain(unsigned *out, int iters, int delta) {
    unsigned a = threadIdx.x + 1;
    unsigned b = threadIdx.x * 7 + 1;
    unsigned c = threadIdx.x * 13 + 1;
    unsigned d = threadIdx.x * 31 + 1;
    for (int i = 0; i < iters; i++) {
        a = __shfl_up_sync(0xffffffff, a, delta);
        b = __shfl_up_sync(0xffffffff, b, delta);
        c = __shfl_up_sync(0xffffffff, c, delta);
        d = __shfl_up_sync(0xffffffff, d, delta);
    }
    if (a + b + c + d == 0xdeadbeef) out[blockIdx.x] = a+b+c+d;
}

__global__ void match_any_chain(unsigned *out, int iters, unsigned val_seed) {
    unsigned a = threadIdx.x ^ val_seed;
    unsigned b = (threadIdx.x * 7) ^ val_seed;
    for (int i = 0; i < iters; i++) {
        a = __match_any_sync(0xffffffff, a + i);
        b = __match_any_sync(0xffffffff, b + i);
    }
    if (a + b == 0xdeadbeef) out[blockIdx.x] = a + b;
}

__global__ void ballot_chain(unsigned *out, int iters, unsigned m) {
    unsigned a = threadIdx.x + 1;
    unsigned b = threadIdx.x * 7 + 1;
    for (int i = 0; i < iters; i++) {
        a = __ballot_sync(0xffffffff, (a + i) > m);
        b = __ballot_sync(0xffffffff, (b + i) > m);
    }
    if (a + b == 0xdeadbeef) out[blockIdx.x] = a + b;
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 1024*sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;
    long total_ops_4 = (long)blocks * threads * iters * 4;
    long total_ops_2 = (long)blocks * threads * iters * 2;

    auto bench = [&](auto launch, long total_ops, const char *name) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double gops = total_ops / (best/1000.0) / 1e9;
        printf("  %-25s %8.3f ms  %8.0f Gops/s\n", name, best, gops);
    };

    printf("# B300 warp shuffle throughput (4-ILP, 148 × 128 threads)\n");
    printf("# %-25s %-12s %-12s\n", "primitive", "time_ms", "Gops/s");

    bench([&]{ shfl_xor_chain<<<blocks, threads>>>(d_out, iters, 1); }, total_ops_4, "shfl.xor mask=1");
    bench([&]{ shfl_xor_chain<<<blocks, threads>>>(d_out, iters, 16); }, total_ops_4, "shfl.xor mask=16");
    bench([&]{ shfl_idx_chain<<<blocks, threads>>>(d_out, iters, 0); }, total_ops_4, "shfl.idx src=0 (bcast)");
    bench([&]{ shfl_idx_chain<<<blocks, threads>>>(d_out, iters, 7); }, total_ops_4, "shfl.idx src=7");
    bench([&]{ shfl_up_chain<<<blocks, threads>>>(d_out, iters, 1); }, total_ops_4, "shfl.up delta=1");
    bench([&]{ match_any_chain<<<blocks, threads>>>(d_out, iters, 0); }, total_ops_2, "match_any");
    bench([&]{ ballot_chain<<<blocks, threads>>>(d_out, iters, 100); }, total_ops_2, "ballot");

    return 0;
}
