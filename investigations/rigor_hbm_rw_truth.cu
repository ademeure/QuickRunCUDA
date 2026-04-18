// Settle HBM R+W full-duplex question via 3 methods + ncu
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void w_v8(int *data) {
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

__launch_bounds__(256, 8) __global__ void r_v8(int *data, int *out) {
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

// SAME kernel does both R AND W (single-kernel mixed)
__launch_bounds__(256, 8) __global__ void rw_v8(int *src, int *dst, int *out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *src_base = src + warp_id * (32 * 1024 / 4);
    int *dst_base = dst + warp_id * (32 * 1024 / 4);
    int acc = 0;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *rp = src_base + (it * 32 + lane) * 8;
        int *wp = dst_base + (it * 32 + lane) * 8;
        int r0,r1,r2,r3,r4,r5,r6,r7;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "l"(rp));
        acc ^= r0 ^ r1 ^ r2 ^ r3;
        asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
            :: "l"(wp), "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7) : "memory");
    }
    if (acc == 0xdeadbeef) out[tid] = acc;
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s_r, s_w;
    cudaStreamCreateWithFlags(&s_r, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_w, cudaStreamNonBlocking);
    cudaEvent_t r0e, r1e, w0e, w1e;
    cudaEventCreate(&r0e); cudaEventCreate(&r1e);
    cudaEventCreate(&w0e); cudaEventCreate(&w1e);

    size_t bytes = 4096ul * 1024 * 1024;
    int *d_a; cudaMalloc(&d_a, bytes); cudaMemset(d_a, 0xab, bytes);
    int *d_b; cudaMalloc(&d_b, bytes);
    int *d_c; cudaMalloc(&d_c, bytes);
    int *d_out; cudaMalloc(&d_out, 16384 * 256 * sizeof(int));
    int blocks = bytes / (256 * 1024), threads = 256;

    // Warmup
    for (int i = 0; i < 3; i++) {
        r_v8<<<blocks, threads, 0, s_r>>>(d_a, d_out);
        w_v8<<<blocks, threads, 0, s_w>>>(d_b);
        rw_v8<<<blocks, threads>>>(d_a, d_c, d_out);
    }
    cudaDeviceSynchronize();

    auto bench_one = [&](auto fn, cudaEvent_t s0, cudaEvent_t s1, cudaStream_t s) {
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(s0, s);
            fn();
            cudaEventRecord(s1, s);
            cudaEventSynchronize(s1);
            float ms; cudaEventElapsedTime(&ms, s0, s1);
            if (ms < best) best = ms;
        }
        return best;
    };

    float t_r = bench_one([&]{ r_v8<<<blocks, threads, 0, s_r>>>(d_a, d_out); }, r0e, r1e, s_r);
    float t_w = bench_one([&]{ w_v8<<<blocks, threads, 0, s_w>>>(d_b); }, w0e, w1e, s_w);
    float t_rw = bench_one([&]{ rw_v8<<<blocks, threads>>>(d_a, d_c, d_out); }, r0e, r1e, 0);

    // Concurrent timing - per stream
    float r_conc = 1e30f, w_conc = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(r0e, s_r);
        r_v8<<<blocks, threads, 0, s_r>>>(d_a, d_out);
        cudaEventRecord(r1e, s_r);
        cudaEventRecord(w0e, s_w);
        w_v8<<<blocks, threads, 0, s_w>>>(d_b);
        cudaEventRecord(w1e, s_w);
        cudaEventSynchronize(r1e);
        cudaEventSynchronize(w1e);
        float ms_r, ms_w;
        cudaEventElapsedTime(&ms_r, r0e, r1e);
        cudaEventElapsedTime(&ms_w, w0e, w1e);
        if (ms_r < r_conc) r_conc = ms_r;
        if (ms_w < w_conc) w_conc = ms_w;
    }

    printf("# Method 1: per-stream wall-clock events\n");
    printf("  Read alone:        %.3f ms = %.0f GB/s\n", t_r, bytes/(t_r/1000)/1e9);
    printf("  Write alone:       %.3f ms = %.0f GB/s\n", t_w, bytes/(t_w/1000)/1e9);
    printf("  Single-kernel R+W: %.3f ms = %.0f R+%.0f W = %.0f GB/s aggregate\n",
           t_rw, bytes/(t_rw/1000)/1e9, bytes/(t_rw/1000)/1e9, 2.0*bytes/(t_rw/1000)/1e9);
    printf("  Concurrent read:   %.3f ms = %.0f GB/s (slowdown %.2fx)\n",
           r_conc, bytes/(r_conc/1000)/1e9, r_conc/t_r);
    printf("  Concurrent write:  %.3f ms = %.0f GB/s (slowdown %.2fx)\n",
           w_conc, bytes/(w_conc/1000)/1e9, w_conc/t_w);
    printf("  Concurrent aggregate (per-stream sum): %.0f GB/s\n",
           bytes/(r_conc/1000)/1e9 + bytes/(w_conc/1000)/1e9);
    return 0;
}
