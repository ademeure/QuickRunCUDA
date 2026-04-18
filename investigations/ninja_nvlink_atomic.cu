// B6: cross-GPU atomic via NVLink — characterize latency + throughput
//
// THEORETICAL:
//   NVLink atomic per prior measurement: ~16 GAtomic/s remote, ~49 GAtomic/s local
//   Latency: catalog says cross-GPU atomic 1.66 us (single round-trip)
//
// Tests:
//   1. Single-thread cross-GPU atomic latency (clock64-based)
//   2. Throughput with N threads concurrent atomicAdd_system to peer
//   3. Compare to local atomicAdd
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__launch_bounds__(32, 1) __global__ void k_one_atom_lat(unsigned long long *p, int *out, int N) {
    long t0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int i = 0; i < N; i++) atomicAdd_system(p, 1ULL);
    long t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) out[0] = (int)((t1 - t0) / N);
}

__global__ void k_many_atom(unsigned long long *p, int N) {
    for (int i = 0; i < N; i++) atomicAdd_system(p, 1ULL);
}

__global__ void k_many_atom_local(unsigned long long *p, int N) {
    for (int i = 0; i < N; i++) atomicAdd(p, 1ULL);
}

int main() {
    cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);

    unsigned long long *d0_remote, *d0_local;
    cudaSetDevice(1); cudaMalloc(&d0_remote, sizeof(unsigned long long));
    cudaMemset(d0_remote, 0, sizeof(unsigned long long));
    cudaSetDevice(0); cudaMalloc(&d0_local, sizeof(unsigned long long));
    cudaMemset(d0_local, 0, sizeof(unsigned long long));

    int *d_out; cudaSetDevice(0); cudaMalloc(&d_out, 32 * sizeof(int));

    // Test 1: single-thread cross-GPU atomic latency via clock64
    cudaSetDevice(0);
    int N = 100;
    for (int i = 0; i < 3; i++) k_one_atom_lat<<<1, 1>>>(d0_remote, d_out, N);
    cudaDeviceSynchronize();
    int cy_per_atom; cudaMemcpy(&cy_per_atom, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    double ns_per_atom = cy_per_atom / 2.032;
    printf("# Single-thread cross-GPU atomicAdd_system latency:\n");
    printf("  %d cy/atom = %.0f ns/atom\n", cy_per_atom, ns_per_atom);

    // Local atomic for comparison
    k_one_atom_lat<<<1, 1>>>(d0_local, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&cy_per_atom, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("# Single-thread LOCAL atomicAdd_system latency:\n");
    printf("  %d cy/atom = %.0f ns/atom\n", cy_per_atom, cy_per_atom/2.032);

    // Test 2: throughput sweep
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    printf("\n# Throughput sweep — N threads doing %d atomicAdd_system to remote\n", N);
    for (int threads : {1, 32, 256, 1024, 1024*32, 1024*148}) {
        int blocks = (threads + 31) / 32;
        int thr_per_block = (threads < 1024) ? threads : 1024;
        if (threads >= 1024) blocks = (threads + 1023) / 1024;
        for (int i = 0; i < 3; i++) k_many_atom<<<blocks, thr_per_block>>>(d0_remote, N);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); k_many_atom<<<blocks, thr_per_block>>>(d0_remote, N); cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total_atoms = (long)threads * N;
        double Gatom_s = total_atoms / (best/1000.0) / 1e9;
        double ns_per_atom_eff = best * 1e6 / total_atoms;
        printf("  threads=%6d  best=%.3f ms  %.2f Gatom/s  %.1f ns/atom\n",
               threads, best, Gatom_s, ns_per_atom_eff);
    }

    printf("\n# LOCAL throughput comparison (no NVLink)\n");
    for (int threads : {1024, 1024*148}) {
        int blocks = (threads + 1023) / 1024;
        int thr_per_block = (threads < 1024) ? threads : 1024;
        for (int i = 0; i < 3; i++) k_many_atom_local<<<blocks, thr_per_block>>>(d0_local, N);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); k_many_atom_local<<<blocks, thr_per_block>>>(d0_local, N); cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total_atoms = (long)threads * N;
        double Gatom_s = total_atoms / (best/1000.0) / 1e9;
        printf("  LOCAL threads=%6d  %.2f Gatom/s\n", threads, Gatom_s);
    }
    return 0;
}
