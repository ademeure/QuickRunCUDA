#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

template <int N> __launch_bounds__(256, 8) __global__ void add32_n(int *p, int N_iters) {
    int idx = (threadIdx.x & 31) % N;
    for (int i = 0; i < N_iters; i++) atomicAdd(&p[idx], 1);
}
template <int N> __launch_bounds__(256, 8) __global__ void xor32_n(int *p, int N_iters) {
    int idx = (threadIdx.x & 31) % N;
    for (int i = 0; i < N_iters; i++) atomicXor(&p[idx], 1);
}
template <int N> __launch_bounds__(256, 8) __global__ void exch32_n(int *p, int N_iters) {
    int idx = (threadIdx.x & 31) % N;
    int v = threadIdx.x;
    for (int i = 0; i < N_iters; i++) v = atomicExch(&p[idx], v + i);
    if (v == 0xdeadbeef) p[64] = v;
}
template <int N> __launch_bounds__(256, 8) __global__ void add64_n(unsigned long long *p, int N_iters) {
    int idx = (threadIdx.x & 31) % N;
    for (int i = 0; i < N_iters; i++) atomicAdd(&p[idx], 1ULL);
}
template <int N> __launch_bounds__(256, 8) __global__ void xor64_n(unsigned long long *p, int N_iters) {
    int idx = (threadIdx.x & 31) % N;
    for (int i = 0; i < N_iters; i++) atomicXor(&p[idx], 1ULL);
}
template <int N> __launch_bounds__(256, 8) __global__ void exch64_n(unsigned long long *p, int N_iters) {
    int idx = (threadIdx.x & 31) % N;
    unsigned long long v = threadIdx.x;
    for (int i = 0; i < N_iters; i++) v = atomicExch(&p[idx], v + i);
    if (v == 0xdeadbeef) p[64] = v;
}
template <int N> __launch_bounds__(256, 8) __global__ void exch128_n(unsigned int *p, int N_iters) {
    int lane = threadIdx.x & 31;
    int idx = (lane % N) * 4;  // 4 ints per b128
    unsigned int v0 = lane, v1 = lane+1, v2 = lane+2, v3 = lane+3;
    unsigned int r0, r1, r2, r3;
    for (int i = 0; i < N_iters; i++) {
        asm volatile(
            "{.reg .b128 d, b;\n"
            "mov.b128 b, {%4, %5, %6, %7};\n"
            "atom.global.b128.exch d, [%8], b;\n"
            "mov.b128 {%0, %1, %2, %3}, d;}\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(v0), "r"(v1), "r"(v2), "r"(v3), "l"(p + idx)
            : "memory"
        );
        v0 = r0 + i;
    }
    if (v0 == 0xdeadbeef) p[256] = v0;
}


int main() {
    cudaSetDevice(0);
    int N_iters = 1000;
    int *d_p32; cudaMalloc(&d_p32, 1024); cudaMemset(d_p32, 0, 1024);
    unsigned long long *d_p64; cudaMalloc(&d_p64, 1024); cudaMemset(d_p64, 0, 1024);
    unsigned int *d_p128; cudaMalloc(&d_p128, 1024); cudaMemset(d_p128, 0, 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run = [&](const char* name, auto kfn, auto p, int width, int n_addr) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(p, N_iters);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(p, N_iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        double T = ops / (best/1000.0) / 1e9;
        double T_per_video_cy = T / 1.860;
        double payload = ops * width / (best/1000.0) / 1e9;
        printf("  %-30s n_addr=%2d  %.3f ms  T=%.1f Gops  T/video-cy=%.2f  payload %.0f GB/s\n",
            name, n_addr, best, T, T_per_video_cy, payload);
    };

    printf("# FULLY CONTENDED: all warps target SAME N addresses (lane %% N)\n\n");
    printf("uint32 ADD:\n");
    run("add32 n=1",     add32_n<1>,  d_p32, 4, 1);
    run("add32 n=4",     add32_n<4>,  d_p32, 4, 4);
    run("add32 n=8 (32B sec)", add32_n<8>, d_p32, 4, 8);
    run("add32 n=16",    add32_n<16>, d_p32, 4, 16);
    run("add32 n=32 (1 line)", add32_n<32>, d_p32, 4, 32);
    printf("\nuint32 XOR:\n");
    run("xor32 n=1",     xor32_n<1>,  d_p32, 4, 1);
    run("xor32 n=8",     xor32_n<8>,  d_p32, 4, 8);
    run("xor32 n=32",    xor32_n<32>, d_p32, 4, 32);
    printf("\nuint32 EXCH:\n");
    run("exch32 n=1",    exch32_n<1>, d_p32, 4, 1);
    run("exch32 n=8",    exch32_n<8>, d_p32, 4, 8);
    run("exch32 n=32",   exch32_n<32>, d_p32, 4, 32);
    printf("\nuint64 ADD:\n");
    run("add64 n=1",     add64_n<1>,  d_p64, 8, 1);
    run("add64 n=4 (32B)", add64_n<4>, d_p64, 8, 4);
    run("add64 n=16 (1 line)", add64_n<16>, d_p64, 8, 16);
    printf("\nuint64 XOR:\n");
    run("xor64 n=1",     xor64_n<1>,  d_p64, 8, 1);
    run("xor64 n=4",     xor64_n<4>,  d_p64, 8, 4);
    run("xor64 n=16",    xor64_n<16>, d_p64, 8, 16);
    printf("\nuint64 EXCH:\n");
    run("exch64 n=1",    exch64_n<1>, d_p64, 8, 1);
    run("exch64 n=4",    exch64_n<4>, d_p64, 8, 4);
    run("exch64 n=16",   exch64_n<16>, d_p64, 8, 16);
    printf("\nb128 EXCH:\n");
    run("b128 exch n=1",  exch128_n<1>,  d_p128, 16, 1);
    run("b128 exch n=2 (32B)", exch128_n<2>, d_p128, 16, 2);
    run("b128 exch n=4 (64B)", exch128_n<4>, d_p128, 16, 4);
    run("b128 exch n=8 (1 line)", exch128_n<8>, d_p128, 16, 8);
    return 0;
}
