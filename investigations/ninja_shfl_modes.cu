// SHFL mode comparison at 1/2/4 SMSPs:
//   .up, .down, .bfly (xor), .idx
// Does .idx (general indexed) have same throughput/scaling as .bfly?
//
// PTX inline syntax:
//   shfl.sync.up.b32   d|p, a, b, c, MASK    (clamp = b above limit)
//   shfl.sync.down.b32 d|p, a, b, c, MASK
//   shfl.sync.bfly.b32 d|p, a, b, c, MASK
//   shfl.sync.idx.b32  d|p, a, b, c, MASK
//
// Use intrinsic forms:
//   __shfl_up_sync, __shfl_down_sync, __shfl_xor_sync, __shfl_sync
#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 64;

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void k_shfl_bfly(int *out, int N_iters) {
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    unsigned v = threadIdx.x ^ 0xCAFE;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            v = __shfl_xor_sync(0xFFFFFFFF, v, j & 31) + j;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void k_shfl_up(int *out, int N_iters) {
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    unsigned v = threadIdx.x ^ 0xCAFE;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            v = __shfl_up_sync(0xFFFFFFFF, v, (j & 31) + 1) + j;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void k_shfl_down(int *out, int N_iters) {
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    unsigned v = threadIdx.x ^ 0xCAFE;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            v = __shfl_down_sync(0xFFFFFFFF, v, (j & 31) + 1) + j;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

// .idx general indexed shuffle — each thread reads from arbitrary src lane
template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void k_shfl_idx(int *out, int N_iters) {
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    int lane = threadIdx.x & 31;
    unsigned v = threadIdx.x ^ 0xCAFE;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int src_lane = (lane + j * 7) & 31;  // arbitrary permutation
            v = __shfl_sync(0xFFFFFFFF, v, src_lane) + j;
        }
    }
    if (v == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = v;
}

template <typename Fn>
double bench(const char* name, Fn kfn, int *d_out, int N_iters, int active_smsp) {
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
    long active_threads = 8L * 32 * active_smsp * 148;
    long inst = active_threads * (long)N_iters * N_INNER / 32;
    double clk_hz = 2032e6;
    double inst_per_sm_per_cy = (double)inst / (best/1000.0) / clk_hz / 148.0;
    printf("  %-25s  %.3f ms  %.2f inst/SM/cy\n", name, best, inst_per_sm_per_cy);
    return inst_per_sm_per_cy;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    int N = 200;
    printf("# SHFL mode scaling — 1, 2, 4 SMSPs each\n");

    printf("# .bfly (XOR):\n");
    bench(".bfly SMSP 0---", k_shfl_bfly<0b0001>, d_out, N, 1);
    bench(".bfly SMSPs 01--", k_shfl_bfly<0b0011>, d_out, N, 2);
    bench(".bfly SMSPs 0123", k_shfl_bfly<0b1111>, d_out, N, 4);

    printf("# .up:\n");
    bench(".up SMSP 0---", k_shfl_up<0b0001>, d_out, N, 1);
    bench(".up SMSPs 01--", k_shfl_up<0b0011>, d_out, N, 2);
    bench(".up SMSPs 0123", k_shfl_up<0b1111>, d_out, N, 4);

    printf("# .down:\n");
    bench(".down SMSP 0---", k_shfl_down<0b0001>, d_out, N, 1);
    bench(".down SMSPs 01--", k_shfl_down<0b0011>, d_out, N, 2);
    bench(".down SMSPs 0123", k_shfl_down<0b1111>, d_out, N, 4);

    printf("# .idx (general indexed):\n");
    bench(".idx SMSP 0---", k_shfl_idx<0b0001>, d_out, N, 1);
    bench(".idx SMSPs 01--", k_shfl_idx<0b0011>, d_out, N, 2);
    bench(".idx SMSPs 0123", k_shfl_idx<0b1111>, d_out, N, 4);
    return 0;
}
