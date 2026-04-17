// Predicate execution cost
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void no_pred(float *out, int iters, int seed) {
    float a = threadIdx.x + 1.0f;
    for (int i = 0; i < iters; i++) {
        a = a * 1.0001f + 0.0001f;
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

extern "C" __global__ void with_pred_uniform(float *out, int iters, int cond) {
    float a = threadIdx.x + 1.0f;
    for (int i = 0; i < iters; i++) {
        if (cond) a = a * 1.0001f + 0.0001f;
        else a = a * 1.0002f + 0.0002f;
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

extern "C" __global__ void with_pred_var(float *out, int iters, int cond_in) {
    float a = threadIdx.x + 1.0f;
    int cond = cond_in;
    for (int i = 0; i < iters; i++) {
        if (cond) a = a * 1.0001f + 0.0001f;
        else a = a * 1.0002f + 0.0002f;
        cond = !cond;  // toggles per iter
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

extern "C" __global__ void with_pred_threadid(float *out, int iters) {
    float a = threadIdx.x + 1.0f;
    int cond = (threadIdx.x & 1);  // half threads take each path
    for (int i = 0; i < iters; i++) {
        if (cond) a = a * 1.0001f + 0.0001f;
        else a = a * 1.0002f + 0.0002f;
    }
    if (a < -1e30f) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 128;

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
        printf("  %-40s %.3f ms\n", name, best);
        return best;
    };

    printf("# B300 predicate execution overhead\n");
    printf("# 148 × 128 thr × 100k iter\n\n");

    bench([&]{ no_pred<<<blocks, threads>>>(d_out, iters, 1); }, "no predicate (baseline)");
    bench([&]{ with_pred_uniform<<<blocks, threads>>>(d_out, iters, 1); }, "if (constant_cond)");
    bench([&]{ with_pred_var<<<blocks, threads>>>(d_out, iters, 0); }, "if (toggling cond)");
    bench([&]{ with_pred_threadid<<<blocks, threads>>>(d_out, iters); }, "if (tid&1) — divergent");

    return 0;
}
