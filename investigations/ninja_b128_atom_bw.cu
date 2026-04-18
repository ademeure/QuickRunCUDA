// 128-bit atomic exchange throughput
#include <cuda_runtime.h>
#include <cstdio>

// 128-bit atomic exch: each thread targets a distinct b128 (16B) slot
// Combine ratio = 8 (8 lanes hit same cache line of 128B)
__launch_bounds__(256, 8) __global__ void k128_combine8(unsigned int *p, int N_iters, long N_int4) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_warps = ((long)gridDim.x * blockDim.x) / 32;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int eighth = lane / 8;
    int sublane = lane & 7;
    unsigned int v0 = tid, v1 = tid+1, v2 = tid+2, v3 = tid+3;
    unsigned int r0, r1, r2, r3;
    for (int i = 0; i < N_iters; i++) {
        // Each "eighth" of warp targets a distinct cache line
        long base_int4 = ((warp_id * 4 + eighth) + (long)i * N_warps * 4) * 8 + sublane;  // 8 int4 per cache line
        long addr_int = (base_int4 % (N_int4)) * 4;  // address in unsigned int
        asm volatile(
            "{\n"
            ".reg .b128 d, b;\n"
            "mov.b128 b, {%4, %5, %6, %7};\n"
            "atom.global.b128.exch d, [%8], b;\n"
            "mov.b128 {%0, %1, %2, %3}, d;\n"
            "}\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(v0), "r"(v1), "r"(v2), "r"(v3), "l"(p + addr_int)
            : "memory"
        );
        // Anti-DCE: feed result back
        v0 = r0; v1 = r1; v2 = r2; v3 = r3;
    }
    if (v0 == 0xdeadbeef) p[0] = v0;
}

// Combine 1: each thread targets distinct cache line
__launch_bounds__(256, 8) __global__ void k128_combine1(unsigned int *p, int N_iters, long N_int4) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    unsigned int v0 = tid, v1 = tid+1, v2 = tid+2, v3 = tid+3;
    unsigned int r0, r1, r2, r3;
    for (int i = 0; i < N_iters; i++) {
        long base_int4 = ((tid + (long)i * N_threads) * 8) % N_int4;  // 8 int4 stride = 1 cache line
        long addr_int = base_int4 * 4;
        asm volatile(
            "{\n.reg .b128 d, b;\n"
            "mov.b128 b, {%4, %5, %6, %7};\n"
            "atom.global.b128.exch d, [%8], b;\n"
            "mov.b128 {%0, %1, %2, %3}, d;\n}\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(v0), "r"(v1), "r"(v2), "r"(v3), "l"(p + addr_int)
            : "memory"
        );
        v0 = r0; v1 = r1; v2 = r2; v3 = r3;
    }
    if (v0 == 0xdeadbeef) p[0] = v0;
}

double bench(const char* name, void(*kfn)(unsigned int*, int, long), unsigned int* d_p, long N_int4) {
    int blocks = 148 * 8, threads = 256, N_iters = 100;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_p, N_iters, N_int4);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s: %s\n", name, cudaGetErrorString(cudaGetLastError())); return 0; }
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        kfn<<<blocks, threads>>>(d_p, N_iters, N_int4);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = (long)blocks * threads * N_iters;
    double gops = ops / (best/1000.0) / 1e9;
    double payload_gbs = ops * 16.0 / (best/1000.0) / 1e9;  // 16 B per atomic
    printf("  %-25s %.3f ms  %.1f Gops/s  payload %.0f GB/s (16B/op)\n",
        name, best, gops, payload_gbs);
    return gops;
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    long WS_MB = (argc > 1) ? atol(argv[1]) : 1024;
    long N_int4 = WS_MB * 1024 * 1024 / 16;  // # of int4 (16B) slots
    long N_int = N_int4 * 4;
    unsigned int *d_p; cudaMalloc(&d_p, (size_t)N_int * 4); cudaMemset(d_p, 0, (size_t)N_int * 4);
    printf("# WS=%ldMB HBM-resident\n", WS_MB);
    bench("b128 exch combine=8", k128_combine8, d_p, N_int4);
    bench("b128 exch combine=1", k128_combine1, d_p, N_int4);
    return 0;
}
