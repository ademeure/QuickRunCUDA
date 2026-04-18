// REDUX/CREDUX with NON-uniform SHFL (compiler can't elide)
//
// Key insight from SASS: SASS has TWO different opcodes:
//   - REDUX.SUM / AND / OR / XOR (one pipe, slower)
//   - CREDUX.MIN / MAX (different pipe, faster)
//
// The "C" prefix in SASS suggests "Compare REDUX" — uses comparator network.
//
// Combo bug fix: SHFL must operate on PER-LANE value, not redux output (uniform).
//   Bad:  v = __shfl_xor_sync(MASK, r, j)   where r = redux output (uniform → SHFL elided)
//   Good: v = __shfl_xor_sync(MASK, v, j) ^ r   where v has per-lane state
#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 64;

__launch_bounds__(1024, 1) __global__ void k_redux_sum(int *out, int N_iters) {
    unsigned v = threadIdx.x ^ 0xCAFE;
    unsigned acc = 0;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            unsigned r;
            asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
            v = (v ^ r) + j;  // per-lane update using r
            acc += r;
        }
    }
    if (acc == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = acc + v;
}

__launch_bounds__(1024, 1) __global__ void k_credux_min(int *out, int N_iters) {
    unsigned v = threadIdx.x ^ 0xCAFE;
    unsigned acc = 0;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            unsigned r;
            asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
            v = (v ^ r) + j;
            acc += r;
        }
    }
    if (acc == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = acc + v;
}

__launch_bounds__(1024, 1) __global__ void k_shfl_only(int *out, int N_iters) {
    unsigned v = threadIdx.x ^ 0xCAFE;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + j;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

// Combo with PROPER per-lane SHFL (cannot be elided)
__launch_bounds__(1024, 1) __global__ void k_redux_sum_shfl(int *out, int N_iters) {
    unsigned v = threadIdx.x ^ 0xCAFE;
    unsigned w = threadIdx.x ^ 0xBEEF;  // separate per-lane state for SHFL
    unsigned acc = 0;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            unsigned r;
            asm volatile("redux.sync.add.s32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
            w = __shfl_xor_sync(0xFFFFFFFF, w, j & 31) ^ r;  // chain redux→SHFL
            v = w + j;  // chain SHFL→redux
            acc += r;
        }
    }
    if (acc == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = acc + v + w;
}

__launch_bounds__(1024, 1) __global__ void k_credux_min_shfl(int *out, int N_iters) {
    unsigned v = threadIdx.x ^ 0xCAFE;
    unsigned w = threadIdx.x ^ 0xBEEF;
    unsigned acc = 0;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            unsigned r;
            asm volatile("redux.sync.min.u32 %0, %1, 0xFFFFFFFF;" : "=r"(r) : "r"(v));
            w = __shfl_xor_sync(0xFFFFFFFF, w, j & 31) ^ r;
            v = w + j;
            acc += r;
        }
    }
    if (acc == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = acc + v + w;
}

template <typename Fn>
double bench(const char* name, Fn kfn, int *d_out, int N_iters, int ops_per_inner) {
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
    long thr_ops = active_threads * (long)N_iters * N_INNER * ops_per_inner;
    long inst = thr_ops / 32;
    double clk_hz = 2032e6;
    double inst_per_sm_per_cy = (double)inst / (best/1000.0) / clk_hz / 148.0;
    printf("  %-25s  %.3f ms  %.2f inst/SM/cy\n", name, best, inst_per_sm_per_cy);
    return inst_per_sm_per_cy;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    int N = 200;
    printf("# Baselines (non-uniform inputs)\n");
    bench("REDUX.SUM alone",   k_redux_sum,        d_out, N, 1);
    bench("CREDUX.MIN alone",  k_credux_min,       d_out, N, 1);
    bench("SHFL alone",        k_shfl_only,        d_out, N, 1);
    printf("\n# Combos (SHFL of per-lane state, not redux output)\n");
    bench("REDUX.SUM + SHFL",  k_redux_sum_shfl,   d_out, N, 2);
    bench("CREDUX.MIN + SHFL", k_credux_min_shfl,  d_out, N, 2);
    return 0;
}
