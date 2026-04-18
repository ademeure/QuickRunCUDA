// Push persistent CPU<->GPU round-trip below 4 us catalog floor
//
// Variants:
//   v1 baseline: ld.acquire.sys + st.release.sys + __sync_synchronize (catalog 3.92 us)
//   v2 relaxed: ld.relaxed.sys + st.relaxed.sys (no fence)
//   v3 weak: regular ld.global.volatile + st.global.volatile (no .sys, no order)
//   v4 atomic: atomic CAS spin
//   v5 weak + no CPU fence
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <pthread.h>
#include <sched.h>

extern "C" __global__ void v1_acq_rel(int *cmd, int *done, int n_iter) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int next = 0;
        while (next < n_iter) {
            int cur;
            do { asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(cur) : "l"(cmd)); } while (cur <= next);
            asm volatile("st.release.sys.u32 [%0], %1;" :: "l"(done), "r"(next + 1));
            next++;
        }
    }
}

extern "C" __global__ void v2_relaxed(int *cmd, int *done, int n_iter) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int next = 0;
        while (next < n_iter) {
            int cur;
            do { asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(cur) : "l"(cmd)); } while (cur <= next);
            asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"(done), "r"(next + 1));
            next++;
        }
    }
}

extern "C" __global__ void v3_volatile(int *cmd, int *done, int n_iter) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        volatile int *vc = cmd;
        volatile int *vd = done;
        int next = 0;
        while (next < n_iter) {
            while (*vc <= next) {}
            *vd = next + 1;
            next++;
        }
    }
}

// CPU side: pin to core
void pin(int core) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

double bench(void(*kfn)(int*, int*, int), const char* name, int *h_cmd, int *h_done, int *d_cmd, int *d_done, int n_work, bool cpu_fence, cudaStream_t s) {
    *h_cmd = 0; *h_done = 0;
    kfn<<<1, 32, 0, s>>>(d_cmd, d_done, n_work);
    // Let kernel start
    while (cudaStreamQuery(s) == cudaSuccess) {}  // wait until kernel running
    // Actually, check kernel is alive by looking at h_cmd value

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_work; i++) {
        volatile int *vh_cmd = (volatile int *)h_cmd;
        volatile int *vh_done = (volatile int *)h_done;
        if (cpu_fence) __sync_synchronize();
        *vh_cmd = i + 1;
        while (*vh_done < i + 1) {}
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    cudaStreamSynchronize(s);
    return us / n_work;
}

int main() {
    cudaSetDevice(0);
    pin(2);  // pin host thread

    int *h_cmd, *h_done;
    cudaHostAlloc(&h_cmd, sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc(&h_done, sizeof(int), cudaHostAllocMapped);
    int *d_cmd, *d_done;
    cudaHostGetDevicePointer(&d_cmd, h_cmd, 0);
    cudaHostGetDevicePointer(&d_done, h_done, 0);
    cudaStream_t s; cudaStreamCreate(&s);

    int N = 10000;
    printf("# Persistent kernel CPU<->GPU round-trip (n_work=%d)\n", N);
    printf("v1 acq.sys/rel.sys + CPU fence: %.3f us/round\n", bench(v1_acq_rel, "v1", h_cmd, h_done, d_cmd, d_done, N, true, s));
    printf("v2 relaxed.sys + CPU fence:     %.3f us/round\n", bench(v2_relaxed, "v2", h_cmd, h_done, d_cmd, d_done, N, true, s));
    printf("v3 volatile (no .sys) + fence:  %.3f us/round\n", bench(v3_volatile, "v3", h_cmd, h_done, d_cmd, d_done, N, true, s));
    printf("v1 acq.sys/rel.sys NO fence:    %.3f us/round\n", bench(v1_acq_rel, "v1", h_cmd, h_done, d_cmd, d_done, N, false, s));
    printf("v2 relaxed.sys NO fence:        %.3f us/round\n", bench(v2_relaxed, "v2", h_cmd, h_done, d_cmd, d_done, N, false, s));
    printf("v3 volatile NO fence:           %.3f us/round\n", bench(v3_volatile, "v3", h_cmd, h_done, d_cmd, d_done, N, false, s));
    return 0;
}
