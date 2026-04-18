// XOR variant + 128-bit exch up to 512B
#include <cuda_runtime.h>
#include <cstdio>

// XOR variants (uint64 atomicXor)
__launch_bounds__(256, 8) __global__ void xor64_all_distinct(unsigned long long *p, int N_iters, long N_addrs) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long N_threads = (long)gridDim.x * blockDim.x;
    for (int i = 0; i < N_iters; i++) {
        long addr = ((tid + (long)i * N_threads) * 16) % N_addrs;
        atomicXor(&p[addr], 1ULL);
    }
}

__launch_bounds__(256, 8) __global__ void xor64_block_4addr(unsigned long long *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx_in_sector = lane & 3;
    long block_base = blockIdx.x * 4;
    long addr = (block_base + target_idx_in_sector) % N_addrs;
    for (int i = 0; i < N_iters; i++) {
        atomicXor(&p[addr], 1ULL);
    }
}

__launch_bounds__(256, 8) __global__ void xor64_block_16addr(unsigned long long *p, int N_iters, long N_addrs) {
    int lane = threadIdx.x & 31;
    int target_idx = lane & 15;
    long block_base = blockIdx.x * 16;
    long addr = (block_base + target_idx) % N_addrs;
    for (int i = 0; i < N_iters; i++) {
        atomicXor(&p[addr], 1ULL);
    }
}

// 128-bit exch with N active lanes (each → 16B distinct addr)
template <int N_ACTIVE>
__launch_bounds__(256, 8) __global__ void b128_exch_n(unsigned int *p, int N_iters) {
    int lane = threadIdx.x & 31;
    if (lane >= N_ACTIVE) return;
    unsigned int v0 = lane, v1 = lane + 1, v2 = lane + 2, v3 = lane + 3;
    unsigned int r0, r1, r2, r3;
    int target_int_offset = lane * 4;  // each b128 = 4 uint = 16B
    for (int i = 0; i < N_iters; i++) {
        asm volatile(
            "{\n"
            ".reg .b128 d, b;\n"
            "mov.b128 b, {%4, %5, %6, %7};\n"
            "atom.global.b128.exch d, [%8], b;\n"
            "mov.b128 {%0, %1, %2, %3}, d;\n"
            "}\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(v0), "r"(v1), "r"(v2), "r"(v3), "l"(p + target_int_offset)
            : "memory"
        );
        v0 = r0 + i; v1 = r1; v2 = r2; v3 = r3;
    }
    if (v0 == 0xdeadbeef) p[256] = v0;
}

int main() {
    cudaSetDevice(0);
    int N_iters = 100;
    long N = 1024L * 1024 * 1024 / 8;
    unsigned long long *d_p; cudaMalloc(&d_p, (size_t)N * 8); cudaMemset(d_p, 0, (size_t)N * 8);
    unsigned int *d_p32; cudaMalloc(&d_p32, 1024 * 1024); cudaMemset(d_p32, 0, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run64 = [&](const char* name, void(*kfn)(unsigned long long*, int, long), int width) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_p, N_iters, N);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_p, N_iters, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        double T = ops / (best/1000.0) / 1e9;
        double T_per_video_cy = T / 1.860;
        double payload = ops * width / (best/1000.0) / 1e9;
        printf("  %-35s %.3f ms  T=%.1f Gops  T/video-cy=%.1f  payload %.0f GB/s\n",
            name, best, T, T_per_video_cy, payload);
    };

    auto run128 = [&](const char* name, void(*kfn)(unsigned int*, int), int n_active) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_p32, N_iters);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_p32, N_iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long active_atomics = (long)blocks * threads * n_active * N_iters / 32;
        double T = active_atomics / (best/1000.0) / 1e9;
        double T_per_video_cy = T / 1.860;
        double payload = active_atomics * 16.0 / (best/1000.0) / 1e9;  // 16B per b128
        printf("  %-35s %.3f ms  T=%.1f Gops  T/video-cy=%.1f  payload %.0f GB/s\n",
            name, best, T, T_per_video_cy, payload);
    };

    printf("# XOR (uint64) variants (HBM-resident WS=1024MB)\n");
    run64("xor64 ALL DISTINCT (1 thread/line)",   xor64_all_distinct, 8);
    run64("xor64 per-block 4 addrs/sector",       xor64_block_4addr,  8);
    run64("xor64 per-block 16 addrs/line",        xor64_block_16addr, 8);

    printf("\n# 128-bit exch with N active lanes (region = N*16B)\n");
    run128("b128 exch 4 lanes (64B = 2 sectors)",  b128_exch_n<4>,  4);
    run128("b128 exch 8 lanes (128B = 1 line)",    b128_exch_n<8>,  8);
    run128("b128 exch 16 lanes (256B = 2 lines)",  b128_exch_n<16>, 16);
    run128("b128 exch 32 lanes (512B = 4 lines)",  b128_exch_n<32>, 32);

    return 0;
}
