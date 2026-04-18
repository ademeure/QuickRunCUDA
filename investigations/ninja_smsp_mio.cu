// MIO pipe unification: do STS, LDS, SHFL, ATOMS all share one per-SM port?
//
// From STS+SHFL combo: 1.0 inst/SM/cy total in any mix (confirmed shared).
// Now verify: ATOMS+SHFL, ATOMS+STS, all 4 simultaneously.
//
// All 4 ops use MIO (Memory Input/Output) pipe per SM in Hopper/Blackwell.
// Theory: MIO can issue 1 op/cy/SM (32 thread-ops via warp-wide instr).
//
// All tests at 4 SMSPs (148 blocks × 1024 threads).
#include <cuda_runtime.h>
#include <cstdio>

constexpr int SMEM_INTS = 32 * 1024;
constexpr int N_INNER = 64;

__launch_bounds__(1024, 1) __global__ void k_atom(int *out, int N_iters) {
    __shared__ int smem[SMEM_INTS];
    for (int s = threadIdx.x; s < SMEM_INTS; s += blockDim.x) smem[s] = 0;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            atomicAdd(&smem[(slot + j * 1024) & (SMEM_INTS-1)], 1);
        }
    }
    if (smem[threadIdx.x] == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = smem[threadIdx.x];
}

__launch_bounds__(1024, 1) __global__ void k_atom_shfl(int *out, int N_iters) {
    __shared__ int smem[SMEM_INTS];
    for (int s = threadIdx.x; s < SMEM_INTS; s += blockDim.x) smem[s] = 0;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    int v = threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            atomicAdd(&smem[(slot + j * 1024) & (SMEM_INTS-1)], 1);
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + i;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

__launch_bounds__(1024, 1) __global__ void k_atom_sts(int *out, int N_iters) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    for (int s = threadIdx.x; s < SMEM_INTS; s += blockDim.x) smem[s] = 0;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            atomicAdd(&smem[(slot + j * 1024) & (SMEM_INTS-1)], 1);
            vsmem[(slot + j * 1024 + 16384) & (SMEM_INTS-1)] = i + j;
        }
    }
}

__launch_bounds__(1024, 1) __global__ void k_atom_sts_shfl(int *out, int N_iters) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    for (int s = threadIdx.x; s < SMEM_INTS; s += blockDim.x) smem[s] = 0;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    int v = threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            atomicAdd(&smem[(slot + j * 1024) & (SMEM_INTS-1)], 1);
            vsmem[(slot + j * 1024 + 16384) & (SMEM_INTS-1)] = i + j;
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + i;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

__launch_bounds__(1024, 1) __global__ void k_lds_shfl(int *out, int N_iters) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    for (int s = threadIdx.x; s < SMEM_INTS; s += blockDim.x) smem[s] = s;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    int v = threadIdx.x, acc = 0;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            acc ^= vsmem[(slot + j * 1024 + i) & (SMEM_INTS-1)];
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + i;
        }
    }
    if (acc == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = acc + v;
}

template <typename Fn>
void bench(const char* name, Fn kfn, int *d_out, int N_iters, int ops_per_inner) {
    int blocks = 148, threads = 1024;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_out, N_iters);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s: %s\n", name, cudaGetErrorString(cudaGetLastError())); return; }
    float best = 1e30f;
    for (int i = 0; i < 8; i++) {
        cudaEventRecord(e0);
        kfn<<<blocks, threads>>>(d_out, N_iters);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long active_threads = (long)blocks * threads;
    long thr_ops = active_threads * (long)N_iters * N_INNER * ops_per_inner;
    long inst = thr_ops / 32;
    double clk_hz = 2032e6;
    double inst_per_sm_per_cy = (double)inst / (best/1000.0) / clk_hz / 148.0;
    double thr_op_per_sm_per_cy = (double)thr_ops / (best/1000.0) / clk_hz / 148.0;
    printf("  %-25s  %.3f ms  %.2f inst/SM/cy  %5.1f thr-op/SM/cy\n",
           name, best, inst_per_sm_per_cy, thr_op_per_sm_per_cy);
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    int N = 200;
    printf("# MIO pipe contention (4 SMSPs)\n");
    bench("ATOMS (.add) only",       k_atom,            d_out, N, 1);
    bench("ATOMS + SHFL",            k_atom_shfl,       d_out, N, 2);
    bench("ATOMS + STS",             k_atom_sts,        d_out, N, 2);
    bench("ATOMS + STS + SHFL",      k_atom_sts_shfl,   d_out, N, 3);
    bench("LDS + SHFL",              k_lds_shfl,        d_out, N, 2);
    return 0;
}
