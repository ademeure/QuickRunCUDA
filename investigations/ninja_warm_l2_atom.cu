// ILP atomic with WARM L2 (small WS that fits in 126 MiB L2)
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void atom_ilp1_warm(unsigned long long *__restrict__ p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long a = ((tid + (long)i * N_threads) * 16) % N_addrs;
        atomicAdd(&p[a], 1ULL);
    }
}

__launch_bounds__(256, 8) __global__ void atom_ilp2_warm(unsigned long long *__restrict__ p1, unsigned long long *__restrict__ p2, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long a = ((tid + (long)i * N_threads) * 16) % N_addrs;
        atomicAdd(&p1[a], 1ULL);
        atomicAdd(&p2[a], 1ULL);
    }
}

__launch_bounds__(256, 8) __global__ void atom_ilp4_warm(
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

int main(int argc, char**argv) {
    cudaSetDevice(0);
    long WS_MB = (argc > 1) ? atol(argv[1]) : 8;  // small WS = warm L2
    long N_addrs = WS_MB * 1024 * 1024 / 8;
    unsigned long long *d_p[4];
    for (int i = 0; i < 4; i++) {
        cudaMalloc(&d_p[i], (size_t)N_addrs * 8);
        cudaMemset(d_p[i], 0, (size_t)N_addrs * 8);
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256, N_iters = 100;

    auto run = [&](const char* name, auto kfn, int n_atom) {
        for (int i = 0; i < 5; i++) kfn();  // warm up L2 first
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 8; i++) {
            cudaEventRecord(e0);
            kfn();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters * n_atom;
        double T = ops / (best/1000.0) / 1e9;
        double payload = ops * 8.0 / (best/1000.0) / 1e9;
        printf("  %-30s WS=%ldMB  %.3f ms  T=%.0f Gops  payload %.0f GB/s\n",
            name, WS_MB, best, T, payload);
    };

    printf("# u64 atomic ILP with warm-L2 WS=%ld MB (vs 126 MiB L2)\n\n", WS_MB);
    run("ILP=1", [&]{atom_ilp1_warm<<<blocks, threads>>>(d_p[0], N_iters, N_addrs);}, 1);
    run("ILP=2", [&]{atom_ilp2_warm<<<blocks, threads>>>(d_p[0], d_p[1], N_iters, N_addrs);}, 2);
    run("ILP=4", [&]{atom_ilp4_warm<<<blocks, threads>>>(d_p[0], d_p[1], d_p[2], d_p[3], N_iters, N_addrs);}, 4);
    return 0;
}
