#include <cuda_runtime.h>
#include <cstdio>

__global__ void w_v8(int *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *base = data + warp_id * (32 * 1024 / 4);
    int v = 0xab;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = base + (it * 32 + lane) * 8;
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}

__global__ void r_v8(int *data, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *base = data + warp_id * (32 * 1024 / 4);
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(p));
        acc ^= r0 ^ r1 ^ r2 ^ r3 ^ r4 ^ r5 ^ r6 ^ r7;
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s_r, s_w;
    cudaStreamCreateWithFlags(&s_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_w, cudaStreamNonBlocking);
    cudaEvent_t r0, r1, w0, w1;
    cudaEventCreate(&r0); cudaEventCreate(&r1);
    cudaEventCreate(&w0); cudaEventCreate(&w1);

    size_t bytes = 4096ul * 1024 * 1024;
    int *d_a; cudaMalloc(&d_a, bytes); cudaMemset(d_a, 0xab, bytes);
    int *d_b; cudaMalloc(&d_b, bytes);
    int *d_out; cudaMalloc(&d_out, 16384 * 256 * sizeof(int));
    int blocks = bytes / (256 * 1024), threads = 256;

    // Warmup
    for (int i = 0; i < 3; i++) {
        r_v8<<<blocks, threads, 0, s_r>>>(d_a, d_out);
        w_v8<<<blocks, threads, 0, s_w>>>(d_b);
    }
    cudaDeviceSynchronize();

    auto bench_alone_r = [&]() {
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(r0, s_r);
            r_v8<<<blocks, threads, 0, s_r>>>(d_a, d_out);
            cudaEventRecord(r1, s_r);
            cudaEventSynchronize(r1);
            float ms; cudaEventElapsedTime(&ms, r0, r1);
            if (ms < best) best = ms;
        }
        return best;
    };
    auto bench_alone_w = [&]() {
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(w0, s_w);
            w_v8<<<blocks, threads, 0, s_w>>>(d_b);
            cudaEventRecord(w1, s_w);
            cudaEventSynchronize(w1);
            float ms; cudaEventElapsedTime(&ms, w0, w1);
            if (ms < best) best = ms;
        }
        return best;
    };

    float t_r = bench_alone_r();
    float t_w = bench_alone_w();

    // Concurrent
    float r_conc_best = 1e30f, w_conc_best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(r0, s_r);
        r_v8<<<blocks, threads, 0, s_r>>>(d_a, d_out);
        cudaEventRecord(r1, s_r);
        cudaEventRecord(w0, s_w);
        w_v8<<<blocks, threads, 0, s_w>>>(d_b);
        cudaEventRecord(w1, s_w);
        cudaEventSynchronize(r1);
        cudaEventSynchronize(w1);
        float ms_r, ms_w;
        cudaEventElapsedTime(&ms_r, r0, r1);
        cudaEventElapsedTime(&ms_w, w0, w1);
        if (ms_r < r_conc_best) r_conc_best = ms_r;
        if (ms_w < w_conc_best) w_conc_best = ms_w;
    }

    printf("# v8 R+W concurrent (per-stream events)\n\n");
    printf("  Read alone:   %.3f ms = %.0f GB/s\n", t_r, bytes/(t_r/1000)/1e9);
    printf("  Write alone:  %.3f ms = %.0f GB/s\n", t_w, bytes/(t_w/1000)/1e9);
    printf("  Concurrent read:  %.3f ms = %.0f GB/s (slowdown %.2fx)\n",
           r_conc_best, bytes/(r_conc_best/1000)/1e9, r_conc_best/t_r);
    printf("  Concurrent write: %.3f ms = %.0f GB/s (slowdown %.2fx)\n",
           w_conc_best, bytes/(w_conc_best/1000)/1e9, w_conc_best/t_w);
    printf("  Aggregate (max):  %.0f GB/s\n",
           bytes/(r_conc_best/1000)/1e9 + bytes/(w_conc_best/1000)/1e9);
    return 0;
}
