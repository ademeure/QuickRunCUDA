// E2 RIGOR: cross-GPU atomic latency dissection
// THEORETICAL: NVLink-5 RTT for B300×2: ~80 ns one-way for small ops.
// Remote atomic = NVLink RTT + remote L2 atomic exec cost.
// Local atomic baseline ~5-10 ns for uncontended scalar add.
// Cross-GPU via P2P should be ~150-200 ns total; the 1.55 us was per
// SOFTWARE round-trip including launches.
//
// Method: single kernel on GPU0 doing N serial atomic ops on REMOTE
// (GPU1) memory via P2P. Measure with clock64 + cudaEvent.

#include <cuda_runtime.h>
#include <cstdio>

#define ITERS 1024

__global__ void chain_atomic_local(unsigned *p, unsigned *clk_out) {
    unsigned tid = threadIdx.x;
    unsigned long long t0 = clock64();
    unsigned acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        acc = atomicAdd(p, 1);
        // Force serial dependency
        if (acc == 0xdeadbeef) p++;
    }
    unsigned long long t1 = clock64();
    if (tid == 0) {
        clk_out[0] = (unsigned)(t1 - t0);
        clk_out[1] = acc;
    }
}

__global__ void chain_atomic_remote(unsigned *remote_p, unsigned *clk_out) {
    unsigned tid = threadIdx.x;
    unsigned long long t0 = clock64();
    unsigned acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        acc = atomicAdd(remote_p, 1);
        if (acc == 0xdeadbeef) remote_p++;
    }
    unsigned long long t1 = clock64();
    if (tid == 0) {
        clk_out[0] = (unsigned)(t1 - t0);
        clk_out[1] = acc;
    }
}

int main() {
    int n_gpus; cudaGetDeviceCount(&n_gpus);
    if (n_gpus < 2) { printf("Need 2 GPUs, found %d\n", n_gpus); return 1; }

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);
    cudaSetDevice(0);

    unsigned *d0_p, *d1_p, *d0_clk;
    cudaMalloc(&d0_p, 1024 * sizeof(unsigned)); cudaMemset(d0_p, 0, 1024 * sizeof(unsigned));
    cudaSetDevice(1);
    cudaMalloc(&d1_p, 1024 * sizeof(unsigned)); cudaMemset(d1_p, 0, 1024 * sizeof(unsigned));
    cudaSetDevice(0);
    cudaMalloc(&d0_clk, 4 * sizeof(unsigned));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    // Local
    chain_atomic_local<<<1, 1>>>(d0_p, d0_clk);  // warmup
    cudaDeviceSynchronize();
    float t_local = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        chain_atomic_local<<<1, 1>>>(d0_p, d0_clk);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < t_local) t_local = ms;
    }
    unsigned local_cycles[2];
    cudaMemcpy(local_cycles, d0_clk, 2*sizeof(unsigned), cudaMemcpyDeviceToHost);

    // Remote
    chain_atomic_remote<<<1, 1>>>(d1_p, d0_clk);  // warmup
    cudaDeviceSynchronize();
    float t_remote = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        chain_atomic_remote<<<1, 1>>>(d1_p, d0_clk);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < t_remote) t_remote = ms;
    }
    unsigned remote_cycles[2];
    cudaMemcpy(remote_cycles, d0_clk, 2*sizeof(unsigned), cudaMemcpyDeviceToHost);

    printf("# Per-op atomic latency (ITERS=%d serial chain)\n\n", ITERS);
    printf("LOCAL  GPU0 atomic on GPU0 mem:\n");
    printf("  wall-clock: %.3f us total = %.1f ns/op\n",
           t_local * 1000, t_local * 1e6 / ITERS);
    printf("  clock64:    %u cycles total = %.1f cy/op (= %.1f ns/op @ 1920 MHz)\n",
           local_cycles[0], (float)local_cycles[0]/ITERS, local_cycles[0]/ITERS / 1.92);

    printf("\nREMOTE GPU0 atomic on GPU1 mem (NVLink P2P):\n");
    printf("  wall-clock: %.3f us total = %.1f ns/op\n",
           t_remote * 1000, t_remote * 1e6 / ITERS);
    printf("  clock64:    %u cycles total = %.1f cy/op (= %.1f ns/op @ 1920 MHz)\n",
           remote_cycles[0], (float)remote_cycles[0]/ITERS, remote_cycles[0]/ITERS / 1.92);

    printf("\n# Dissection:\n");
    double local_ns = t_local * 1e6 / ITERS;
    double remote_ns = t_remote * 1e6 / ITERS;
    printf("  Local atomic cost      : %.1f ns\n", local_ns);
    printf("  Remote atomic cost     : %.1f ns\n", remote_ns);
    printf("  Cross-GPU overhead     : %.1f ns (NVLink RTT + protocol)\n", remote_ns - local_ns);
    printf("  Ratio remote/local     : %.1fx\n", remote_ns / local_ns);

    return 0;
}
