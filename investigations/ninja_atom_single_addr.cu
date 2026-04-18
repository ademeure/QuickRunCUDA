// All SMs atomic-add to SINGLE address — measure single-cache-line atomic throughput
#include <cuda_runtime.h>
#include <cstdio>

// All threads target ADDRESS 0 (single int)
__launch_bounds__(256, 8) __global__ void atom_one_addr(int *p, int N_iters) {
    for (int i = 0; i < N_iters; i++) {
        atomicAdd(&p[0], 1);
    }
}

// All threads target FIRST 4 bytes (still 1 cache line, 1 int)
// vs DIFFERENT 16-byte b128 atomic on same line
__launch_bounds__(256, 8) __global__ void atom_one_addr_b128(int *p, int N_iters) {
    unsigned int v0 = 1, v1 = 2, v2 = 3, v3 = 4;
    unsigned int r0, r1, r2, r3;
    for (int i = 0; i < N_iters; i++) {
        asm volatile(
            "{\n"
            ".reg .b128 d, b;\n"
            "mov.b128 b, {%4, %5, %6, %7};\n"
            "atom.global.b128.exch d, [%8], b;\n"
            "mov.b128 {%0, %1, %2, %3}, d;\n"
            "}\n"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(v0), "r"(v1), "r"(v2), "r"(v3), "l"(p)
            : "memory"
        );
        v0 = r0; v1 = r1; v2 = r2; v3 = r3;
    }
    if (v0 == 0xdeadbeef) p[0] = v0;
}

int main(int argc, char**argv) {
    cudaSetDevice(0);
    int N_iters = (argc > 1) ? atoi(argv[1]) : 1000;
    int *d_p; cudaMalloc(&d_p, 1024);
    cudaMemset(d_p, 0, 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto run = [&](const char* name, void(*kfn)(int*, int), int blocks, int threads, int width_bytes) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_p, N_iters);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", name); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_p, N_iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        double gops_thread = ops / (best/1000.0) / 1e9;
        // For single-address atomic: combine=32 within warp → 1 L2 packet per warp
        double gops_warp = gops_thread / 32;
        double payload_gbs = ops * width_bytes / (best/1000.0) / 1e9;
        // Per video-clock cy
        double l2_pkt_per_video_cy = gops_warp / 1.860;  // video clock 1.860 GHz
        printf("  %-25s blocks=%d thr=%d  %.3f ms  T=%.1f Gops  W=%.2f Gwarp/s  L2pkt/cy=%.2f  payload=%.0f GB/s\n",
            name, blocks, threads, best, gops_thread, gops_warp, l2_pkt_per_video_cy, payload_gbs);
    };

    printf("# All SMs atomic-add to SINGLE address (max contention test)\n");
    printf("# 32 lanes per warp target same address -> 1 L2 packet per warp\n");
    run("int32 ALL same addr",  atom_one_addr,      148*8, 256, 4);
    run("int32 ALL same addr (1 block)", atom_one_addr, 1, 256, 4);
    run("int32 ALL same addr (148 blk)", atom_one_addr, 148, 256, 4);
    run("int32 ALL same addr (296 blk)", atom_one_addr, 296, 256, 4);
    run("int32 ALL same addr (592 blk)", atom_one_addr, 592, 256, 4);
    run("b128 exch ALL same addr", atom_one_addr_b128, 148*8, 256, 16);

    return 0;
}
