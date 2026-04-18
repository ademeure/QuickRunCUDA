// SMEM STS per-SMSP scaling — fix v3
//
// v2 bugs:
//   (1) Address pattern (w*32+l)*32 → all 32 lanes hit same bank → 32-way conflict
//   (2) DCE elided the chain
//
// v3 fixes:
//   - Address pattern: slot = warp_id*32 + lane → lane l → bank l (conflict-free)
//   - Use volatile to prevent compiler from removing stores
//   - Time-scaling check: rate per N_iters should be constant
#include <cuda_runtime.h>
#include <cstdio>

constexpr int SMEM_INTS = 32 * 1024;
constexpr int N_INNER = 64;

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void sts_only_v(int *out, int N_iters, int seed) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;  // conflict-free: lane l → bank l
    int val = threadIdx.x ^ seed;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 1024) & (SMEM_INTS - 1);
            vsmem[s] = val + i + j;
        }
    }
}

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void lds_only_v(int *out, int N_iters, int seed) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    for (int s = threadIdx.x; s < SMEM_INTS; s += blockDim.x) smem[s] = s ^ seed;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    int acc = 0;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 1024 + i) & (SMEM_INTS - 1);
            acc ^= vsmem[s];
        }
    }
    if (acc == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = acc;
}

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void stsd_lds_v(int *out, int N_iters, int seed) {
    __shared__ int smem[SMEM_INTS];
    volatile int *vsmem = smem;
    for (int s = threadIdx.x; s < SMEM_INTS; s += blockDim.x) smem[s] = s ^ seed;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    int acc = 0;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 1024 + i) & (SMEM_INTS - 1);
            vsmem[s] = i + j + threadIdx.x;
            acc ^= vsmem[s];
        }
    }
    if (acc == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = acc;
}

template <typename Fn>
double bench(const char* name, Fn kfn, int *d_out, int N_iters, int active_smsp, int ops_per_inner) {
    int blocks = 148, threads = 1024;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_out, N_iters, 0xCAFE);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s: %s\n", name, cudaGetErrorString(cudaGetLastError())); return 0; }
    float best = 1e30f;
    for (int i = 0; i < 8; i++) {
        cudaEventRecord(e0);
        kfn<<<blocks, threads>>>(d_out, N_iters, 0xCAFE + i);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long active_threads = 8L * 32 * active_smsp * 148;
    long thr_ops = active_threads * (long)N_iters * N_INNER * ops_per_inner;
    long warp_inst = thr_ops / 32;
    double clk_hz = 2032e6;
    double thr_op_per_sm_per_cy = (double)thr_ops / (best/1000.0) / clk_hz / 148.0;
    double inst_per_sm_per_cy = (double)warp_inst / (best/1000.0) / clk_hz / 148.0;
    double bytes_per_sec = (double)thr_ops * 4 / (best/1000.0);
    printf("  %-30s  %.3f ms  %5.0f Gthr-op  %5.1f thr-op/SM/cy  %5.2f inst/SM/cy  %.2f TB/s\n",
           name, best, (double)thr_ops/(best/1000.0)/1e9, thr_op_per_sm_per_cy, inst_per_sm_per_cy, bytes_per_sec/1e12);
    return inst_per_sm_per_cy;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));

    printf("# Time-scaling sanity (rate should be invariant across N_iters)\n");
    bench("STS-only N=100", sts_only_v<0b1111>, d_out, 100, 4, 1);
    bench("STS-only N=200", sts_only_v<0b1111>, d_out, 200, 4, 1);
    bench("STS-only N=400", sts_only_v<0b1111>, d_out, 400, 4, 1);

    printf("\n# === STS-only — single SMSP ===\n");
    bench("STS-only SMSP 0---",  sts_only_v<0b0001>, d_out, 200, 1, 1);
    bench("STS-only SMSP -1--",  sts_only_v<0b0010>, d_out, 200, 1, 1);
    bench("STS-only SMSP --2-",  sts_only_v<0b0100>, d_out, 200, 1, 1);
    bench("STS-only SMSP ---3",  sts_only_v<0b1000>, d_out, 200, 1, 1);

    printf("\n# === STS-only — pairs ===\n");
    bench("STS-only SMSPs 01--", sts_only_v<0b0011>, d_out, 200, 2, 1);
    bench("STS-only SMSPs 0-2-", sts_only_v<0b0101>, d_out, 200, 2, 1);
    bench("STS-only SMSPs 0--3", sts_only_v<0b1001>, d_out, 200, 2, 1);
    bench("STS-only SMSPs -12-", sts_only_v<0b0110>, d_out, 200, 2, 1);
    bench("STS-only SMSPs -1-3", sts_only_v<0b1010>, d_out, 200, 2, 1);
    bench("STS-only SMSPs --23", sts_only_v<0b1100>, d_out, 200, 2, 1);

    printf("\n# === STS-only — triples & quad ===\n");
    bench("STS-only SMSPs 012-", sts_only_v<0b0111>, d_out, 200, 3, 1);
    bench("STS-only SMSPs 01-3", sts_only_v<0b1011>, d_out, 200, 3, 1);
    bench("STS-only SMSPs 0-23", sts_only_v<0b1101>, d_out, 200, 3, 1);
    bench("STS-only SMSPs -123", sts_only_v<0b1110>, d_out, 200, 3, 1);
    bench("STS-only SMSPs 0123", sts_only_v<0b1111>, d_out, 200, 4, 1);

    printf("\n# === LDS-only (re-verify) ===\n");
    bench("LDS-only SMSP 0---", lds_only_v<0b0001>, d_out, 200, 1, 1);
    bench("LDS-only SMSPs 01--", lds_only_v<0b0011>, d_out, 200, 2, 1);
    bench("LDS-only SMSPs 0-2-", lds_only_v<0b0101>, d_out, 200, 2, 1);
    bench("LDS-only SMSPs 0123", lds_only_v<0b1111>, d_out, 200, 4, 1);

    printf("\n# === STS+LDS interleaved (alt) ===\n");
    bench("STS+LDS SMSP 0---", stsd_lds_v<0b0001>, d_out, 200, 1, 2);
    bench("STS+LDS SMSPs 01--", stsd_lds_v<0b0011>, d_out, 200, 2, 2);
    bench("STS+LDS SMSPs 0123", stsd_lds_v<0b1111>, d_out, 200, 4, 2);

    return 0;
}
