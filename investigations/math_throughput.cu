// Math function throughput on B300 (MUFU and friends)
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <chrono>

#define ITERS 50000
#define ILP 8

#define BENCH_FUNC(NAME, FN) \
extern "C" __global__ void k_##NAME(float *in, float *out) { \
    float a[ILP]; \
    for (int i = 0; i < ILP; i++) a[i] = in[i]; \
    _Pragma("unroll 1") \
    for (int i = 0; i < ITERS; i++) { \
        _Pragma("unroll") \
        for (int j = 0; j < ILP; j++) a[j] = FN(a[j]) + 0.5f; \
    } \
    if (threadIdx.x == 0) { \
        float s = 0; \
        for (int i = 0; i < ILP; i++) s += a[i]; \
        out[blockIdx.x] = s; \
    } \
}

BENCH_FUNC(sqrt, sqrtf)
BENCH_FUNC(rsqrt, rsqrtf)
BENCH_FUNC(rcp, __frcp_rn)
BENCH_FUNC(sin, __sinf)
BENCH_FUNC(cos, __cosf)
BENCH_FUNC(tan, __tanf)
BENCH_FUNC(exp, __expf)
BENCH_FUNC(log, __logf)
BENCH_FUNC(exp2, exp2f)
BENCH_FUNC(log2, log2f)

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    float *d_in, *d_out;
    cudaMalloc(&d_in, 16 * sizeof(float));
    cudaMalloc(&d_out, sm * sizeof(float));

    float h_in[16];
    for (int i = 0; i < 16; i++) h_in[i] = 0.5f + i * 0.1f;
    cudaMemcpy(d_in, h_in, 16 * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    auto run = [&](const char *name, void (*fn_ptr)(float*, float*)) {
        float t = bench([&]{
            fn_ptr<<<sm, 128, 0, s>>>(d_in, d_out);
        });
        long long ops = (long long)sm * 128 * ITERS * ILP;
        double tops = (double)ops / (t/1e3) / 1e12;
        printf("  %-12s : %.3f ms, %.2f Gops/s aggregate (%.2f Gops/s/SM)\n",
               name, t, tops*1e3, tops*1e3/sm);
    };

    printf("# B300 math function throughput\n");
    printf("# 148 blocks × 128 threads × ILP=%d × %d iters\n\n", ILP, ITERS);

    run("sqrtf", k_sqrt);
    run("rsqrtf", k_rsqrt);
    run("__frcp_rn", k_rcp);
    run("__sinf", k_sin);
    run("__cosf", k_cos);
    run("__tanf", k_tan);
    run("__expf", k_exp);
    run("__logf", k_log);
    run("exp2f", k_exp2);
    run("log2f", k_log2);

    cudaStreamDestroy(s);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
