// Per-L2-unit throughput: warm vs cold, single-line full contention
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void atom_one_addr(int *p, int N_iters) {
    for (int i = 0; i < N_iters; i++) atomicAdd(&p[0], 1);  // popc-merge same addr
}

__launch_bounds__(256, 8) __global__ void exch_one_addr(int *p, int N_iters) {
    int v = threadIdx.x;
    for (int i = 0; i < N_iters; i++) v = atomicExch(&p[0], v + i);  // no popc-merge
    if (v == 0xdeadbeef) p[1] = v;
}

int main() {
    cudaSetDevice(0);
    int N_iters = 1000;
    // Tiny allocation (single int) - always L2-warm after first touch
    int *d_p_warm; cudaMalloc(&d_p_warm, 64);
    cudaMemset(d_p_warm, 0, 64);
    // Sandwich a HBM-bound noise allocation to evict L2 BEFORE cold test
    char *d_noise; cudaMalloc(&d_noise, 256*1024*1024);  // 256MB noise
    cudaMemset(d_noise, 0xa5, 256*1024*1024);
    // Cold target = single int but L2 evicted
    int *d_p_cold; cudaMalloc(&d_p_cold, 64);
    cudaMemset(d_p_cold, 0, 64);
    
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;

    auto run = [&](const char* name, void(*kfn)(int*, int), int *d_p, bool evict_first) {
        // Warmup runs to get target into L2
        for (int i = 0; i < 5; i++) kfn<<<blocks, threads>>>(d_p, N_iters);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR\n"); return; }
        if (evict_first) {
            // Evict L2 by reading huge area
            cudaMemset(d_noise, 0xa6, 256*1024*1024);
            cudaDeviceSynchronize();
        }
        float best = 1e30f;
        for (int trial = 0; trial < 8; trial++) {
            if (evict_first) {
                cudaMemset(d_noise, 0xa7 + trial, 256*1024*1024);
                cudaDeviceSynchronize();
            }
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_p, N_iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * threads * N_iters;
        long warps = (long)blocks * threads / 32 * N_iters;
        double T = ops / (best/1000.0) / 1e9;
        double W = warps / (best/1000.0) / 1e9;
        double per_warp_cy = 1e9 / W * 1.860;  // video cy per warp
        printf("  %-30s %s  %.3f ms  T=%.1f Gops  W=%.2f Gwarp/s  per-warp=%.2f video-cy\n",
            name, evict_first ? "(evict)" : "(warm) ", best, T, W, per_warp_cy);
    };

    printf("# Single-line atomic: WARM L2 vs COLD (L2-evicted)\n");
    printf("# All %d blocks * 256 thr -> address [0]\n\n", blocks);
    run("ADD same-addr WARM",  atom_one_addr, d_p_warm, false);
    run("ADD same-addr COLD",  atom_one_addr, d_p_cold, true);
    run("EXCH same-addr WARM", exch_one_addr, d_p_warm, false);
    run("EXCH same-addr COLD", exch_one_addr, d_p_cold, true);

    return 0;
}
