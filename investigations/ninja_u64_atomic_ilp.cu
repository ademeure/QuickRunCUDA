// ILP atomic: 2x or 4x b64 per thread, addresses GB apart for true HBM saturation
#include <cuda_runtime.h>
#include <cstdio>

// 1x: baseline - one atomic per thread per iter
__launch_bounds__(256, 8) __global__ void atom_ilp1(unsigned long long *__restrict__ p1, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long a1 = ((tid + (long)i * N_threads) * 16) % N_addrs;
        atomicAdd(&p1[a1], 1ULL);
    }
}

// 2x ILP: 2 distinct atomics per thread, addresses GB apart
__launch_bounds__(256, 8) __global__ void atom_ilp2(unsigned long long *__restrict__ p1, unsigned long long *__restrict__ p2, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long a1 = ((tid + (long)i * N_threads) * 16) % N_addrs;
        long a2 = ((tid + (long)i * N_threads) * 16) % N_addrs;
        atomicAdd(&p1[a1], 1ULL);
        atomicAdd(&p2[a2], 1ULL);
    }
}

// 4x ILP: 4 distinct atomics per thread, different GB regions
__launch_bounds__(256, 8) __global__ void atom_ilp4(
    unsigned long long *__restrict__ p1, unsigned long long *__restrict__ p2,
    unsigned long long *__restrict__ p3, unsigned long long *__restrict__ p4,
    int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long a = ((tid + (long)i * N_threads) * 16) % N_addrs;
        atomicAdd(&p1[a], 1ULL);
        atomicAdd(&p2[a], 1ULL);
        atomicAdd(&p3[a], 1ULL);
        atomicAdd(&p4[a], 1ULL);
    }
}

// b128 4x ILP version
__launch_bounds__(256, 8) __global__ void exch128_ilp4(
    unsigned int *__restrict__ p1, unsigned int *__restrict__ p2,
    unsigned int *__restrict__ p3, unsigned int *__restrict__ p4,
    int N_iters, long N_addrs_int) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    unsigned int v0 = tid, v1 = tid+1, v2 = tid+2, v3 = tid+3;
    for (int i = 0; i < N_iters; i++) {
        long a = (((tid + (long)i * N_threads) * 32)) % N_addrs_int;
        unsigned int r0, r1, r2, r3;
        // 4 b128 atomics in parallel
        for (int j = 0; j < 4; j++) {
            unsigned int *p = (j == 0 ? p1 : (j == 1 ? p2 : (j == 2 ? p3 : p4)));
            asm volatile(
                "{.reg .b128 d, b;\n"
                "mov.b128 b, {%4, %5, %6, %7};\n"
                "atom.global.b128.exch d, [%8], b;\n"
                "mov.b128 {%0, %1, %2, %3}, d;}\n"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                : "r"(v0), "r"(v1), "r"(v2), "r"(v3), "l"(p + a)
                : "memory"
            );
        }
        v0 = r0;
    }
    if (v0 == 0xdeadbeef) p1[0] = v0;
}

int main() {
    cudaSetDevice(0);
    long WS_GB = 2;  // each region 2 GB
    long N_addrs = WS_GB * 1024 * 1024 * 1024 / 8;
    long N_addrs_int = WS_GB * 1024 * 1024 * 1024 / 4;
    unsigned long long *d_p1, *d_p2, *d_p3, *d_p4;
    cudaMalloc(&d_p1, (size_t)N_addrs * 8);
    cudaMalloc(&d_p2, (size_t)N_addrs * 8);
    cudaMalloc(&d_p3, (size_t)N_addrs * 8);
    cudaMalloc(&d_p4, (size_t)N_addrs * 8);
    cudaMemset(d_p1, 0, (size_t)N_addrs * 8);
    cudaMemset(d_p2, 0, (size_t)N_addrs * 8);
    cudaMemset(d_p3, 0, (size_t)N_addrs * 8);
    cudaMemset(d_p4, 0, (size_t)N_addrs * 8);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run = [&](const char* name, auto kfn, int n_atomics_per_iter, int width) {
        for (int i = 0; i < 3; i++) kfn();
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * 100 * n_atomics_per_iter;
        double T = ops / (best/1000.0) / 1e9;
        double payload = ops * width / (best/1000.0) / 1e9;
        printf("  %-25s %.3f ms  T=%.0f Gops  payload %.0f GB/s = %.2f TB/s\n",
            name, best, T, payload, payload/1000);
    };

    int N_iters = 100;
    printf("# u64/b128 ILP atomic, addresses GB apart, 2048 thr/SM\n\n");
    run("u64 ILP=1 (baseline)",
        [&]{atom_ilp1<<<blocks, threads>>>(d_p1, N_iters, N_addrs);}, 1, 8);
    run("u64 ILP=2 (2 GB apart)",
        [&]{atom_ilp2<<<blocks, threads>>>(d_p1, d_p2, N_iters, N_addrs);}, 2, 8);
    run("u64 ILP=4 (4 GB apart)",
        [&]{atom_ilp4<<<blocks, threads>>>(d_p1, d_p2, d_p3, d_p4, N_iters, N_addrs);}, 4, 8);
    run("b128 ILP=4",
        [&]{exch128_ilp4<<<blocks, threads>>>((unsigned int*)d_p1, (unsigned int*)d_p2, (unsigned int*)d_p3, (unsigned int*)d_p4, N_iters, N_addrs_int);}, 4, 16);
    return 0;
}
