// Integer ops with anti-DCE (runtime constants from kernel arg)
#include <cuda_runtime.h>
#include <cstdio>

template<typename Op>
__global__ void run_int(unsigned *out, int iters, unsigned k1, unsigned k2, Op op) {
    unsigned a = threadIdx.x + 1;
    unsigned b = threadIdx.x * 7 + 1;
    unsigned c = threadIdx.x * 13 + 1;
    unsigned d = threadIdx.x * 31 + 1;
    for (int i = 0; i < iters; i++) {
        a = op(a, k1, k2); b = op(b, k1, k2); c = op(c, k1, k2); d = op(d, k1, k2);
    }
    if (a + b + c + d == 0xdeadbeef) out[blockIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 1024*sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;
    long total_ops = (long)blocks * threads * iters * 4;
    unsigned k1 = 17, k2 = 13;  // runtime kernel args

    auto bench = [&](auto launch, const char *name) {
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
        printf("  %-20s %8.3f ms  %8.1f Gops/s\n", name, best, gops);
    };

    printf("# B300 integer ops with runtime k1, k2 (defeats DCE)\n");
    printf("# 4-ILP chains, 148 × 128 = 18944 threads × 100k iter × 4\n");
    printf("# %-20s %-12s %-12s\n", "op", "time_ms", "Gops/s");

    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return x + a; }); }, "iadd");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return x * a; }); }, "imul");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return __umul24(x, a); }); }, "umul24");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return x * a + b; }); }, "imad");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return (x << (a&31)) | (x >> ((32-a)&31)); }); }, "rotl (var)");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return x & a; }); }, "and");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return x ^ a; }); }, "xor");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return __popc(x ^ a); }); }, "popc");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return __clz(x | a); }); }, "clz");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return __brev(x ^ a); }); }, "brev");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return __byte_perm(x, a, 0x4321); }); }, "byte_perm");
    bench([&]{ run_int<<<blocks, threads>>>(d_out, iters, k1, k2,
        [] __device__ (unsigned x, unsigned a, unsigned b) { return x / a + b; }); }, "udiv");

    return 0;
}
