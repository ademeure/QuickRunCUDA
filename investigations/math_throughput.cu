// Math intrinsic throughput: rsqrt, sqrt, sin, cos, exp, log, ex2, lg2
#include <cuda_runtime.h>
#include <cstdio>

template<typename Op>
__global__ void run_op(float *out, int iters, Op op) {
    float a = 1.0f + threadIdx.x * 0.001f;
    float b = 1.5f + threadIdx.x * 0.0001f;
    float c = 2.0f + threadIdx.x * 0.0002f;
    float d = 2.5f + threadIdx.x * 0.0003f;
    for (int i = 0; i < iters; i++) {
        a = op(a); b = op(b); c = op(c); d = op(d);
    }
    if (a + b + c + d < -1e30f) out[blockIdx.x*blockDim.x + threadIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 148*256*sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;
    long total_ops = (long)blocks * threads * iters * 4;  // 4 chains

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
        printf("  %-15s %8.3f ms  %8.1f Gops/s  ratio_to_FMA=%.2fx\n",
               name, best, gops, gops);
    };

    printf("# B300 math throughput (4 ILP chains, 128 thr/block, 148 blocks)\n");
    printf("# All measurements: G float-ops per second\n\n");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return x * 1.0001f + 0.0001f; }); }, "FMA(baseline)");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return __frsqrt_rn(x); }); }, "rsqrt");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return sqrtf(x); }); }, "sqrt");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return __sinf(x); }); }, "sin (intrin)");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return __cosf(x); }); }, "cos (intrin)");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return __expf(x); }); }, "exp (intrin)");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return __logf(x); }); }, "log (intrin)");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return exp2f(x); }); }, "exp2");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return log2f(x); }); }, "log2");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return tanhf(x); }); }, "tanhf");

    bench([&]{ run_op<<<blocks, threads>>>(d_out, iters,
        [] __device__ (float x) { return 1.0f / x; }); }, "div");

    return 0;
}
