// SMEM STS + SHFL contention — are they truly independent units?
//
// At 4 SMSPs:
//   STS-only:  0.98 inst/SM/cy = 32 thr-op/SM/cy (1 SMEM port)
//   SHFL-only: 1.00 inst/SM/cy = 32 thr-op/SM/cy (cross-bar)
//
// If both INDEPENDENT (separate units): combined → 32 STS + 32 SHFL per cy per SM
//   → 64 thr-op/SM/cy aggregate
// If SHARED issue slot (1 inst/cy/SMSP): combined → 32 thr-op/SM/cy total
//   → kernel time scales linearly with mix
// If SHARED other resource: somewhere between
//
// Test: kernel does N_INNER ops alternating STS+SHFL. Compare to STS-only + SHFL-only.
#include <cuda_runtime.h>
#include <cstdio>

constexpr int SMEM_INTS = 32 * 1024;
constexpr int N_INNER = 64;

// Pure STS reference
__launch_bounds__(1024, 1) __global__ void k_sts(int *out, int N_iters) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    int val = threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 1024) & (SMEM_INTS - 1);
            vsmem[s] = val + i + j;
        }
    }
}

// Pure SHFL reference
__launch_bounds__(1024, 1) __global__ void k_shfl(int *out, int N_iters) {
    int v = threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + i;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

// Combined STS + SHFL alternating (1:1)
__launch_bounds__(1024, 1) __global__ void k_sts_shfl(int *out, int N_iters) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    int v = threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 1024) & (SMEM_INTS - 1);
            vsmem[s] = v + i + j;
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + i;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

// 2 STS + 1 SHFL
__launch_bounds__(1024, 1) __global__ void k_2sts_1shfl(int *out, int N_iters) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    int v = threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s1 = (slot + j * 1024) & (SMEM_INTS - 1);
            int s2 = (slot + j * 1024 + 32) & (SMEM_INTS - 1);
            vsmem[s1] = v + i;
            vsmem[s2] = v + j;
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + i;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

// 1 STS + 2 SHFL
__launch_bounds__(1024, 1) __global__ void k_1sts_2shfl(int *out, int N_iters) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    int v = threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 1024) & (SMEM_INTS - 1);
            vsmem[s] = v + i + j;
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + i;
            v = __shfl_xor_sync(0xFFFFFFFF, v, (j + 5) & 31) + j;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

template <typename Fn>
double bench(const char* name, Fn kfn, int *d_out, int N_iters, int sts_per_inner, int shfl_per_inner) {
    int blocks = 148, threads = 1024;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_out, N_iters);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s: %s\n", name, cudaGetErrorString(cudaGetLastError())); return 0; }
    float best = 1e30f;
    for (int i = 0; i < 8; i++) {
        cudaEventRecord(e0);
        kfn<<<blocks, threads>>>(d_out, N_iters);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long active_threads = (long)blocks * threads;
    long sts_thr_ops = active_threads * (long)N_iters * N_INNER * sts_per_inner;
    long shfl_thr_ops = active_threads * (long)N_iters * N_INNER * shfl_per_inner;
    long total_thr_ops = sts_thr_ops + shfl_thr_ops;
    double clk_hz = 2032e6;
    double sts_per_sm_per_cy = (double)sts_thr_ops / (best/1000.0) / clk_hz / 148.0 / 32.0;  // inst
    double shfl_per_sm_per_cy = (double)shfl_thr_ops / (best/1000.0) / clk_hz / 148.0 / 32.0;
    double total_per_sm_per_cy = (double)total_thr_ops / (best/1000.0) / clk_hz / 148.0;  // thr-ops
    printf("  %-25s  %.3f ms  STS=%.2f inst/SM/cy  SHFL=%.2f inst/SM/cy  total=%.1f thr-op/SM/cy\n",
           name, best, sts_per_sm_per_cy, shfl_per_sm_per_cy, total_per_sm_per_cy);
    return total_per_sm_per_cy;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    int N = 200;
    printf("# STS+SHFL contention (4 SMSPs all active)\n");
    bench("STS-only",   k_sts,        d_out, N, 1, 0);
    bench("SHFL-only",  k_shfl,       d_out, N, 0, 1);
    bench("1 STS + 1 SHFL", k_sts_shfl,    d_out, N, 1, 1);
    bench("2 STS + 1 SHFL", k_2sts_1shfl,  d_out, N, 2, 1);
    bench("1 STS + 2 SHFL", k_1sts_2shfl,  d_out, N, 1, 2);
    return 0;
}
