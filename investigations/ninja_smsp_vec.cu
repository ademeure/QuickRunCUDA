// SMEM .32 vs .64 vs .128 vector STS/LDS — does wider load saturate port faster?
//
// Hypothesis A (1 inst/cy port):
//   .32 → 128B/cy/SM, .64 → 256B/cy/SM, .128 → 512B/cy/SM
//   Then 1 SMSP doing .128 alone might saturate the port.
//
// Hypothesis B (128B/cy port, wider just multi-cycle):
//   .128 takes 4 cy per inst → same BW as .32, just fewer instructions
//
// Test: bench .32 / .64 / .128 STS/LDS at 1, 2, 4 active SMSPs.
// Compute: bytes/sec achieved. Compare to 38.5 TB/s catalog max.
#include <cuda_runtime.h>
#include <cstdio>

constexpr int SMEM_BYTES = 128 * 1024;  // 128 KB per SM SMEM
constexpr int N_INNER = 32;

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void sts32(int *out, int N_iters) {
    __shared__ int smem[SMEM_BYTES / 4];
    volatile int *vsmem = smem;
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    int val = threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 1024) & (SMEM_BYTES/4 - 1);
            vsmem[s] = val + i + j;
        }
    }
}

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void sts64(unsigned long long *out, int N_iters) {
    __shared__ unsigned long long smem[SMEM_BYTES / 8];
    volatile unsigned long long *vsmem = smem;
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    unsigned long long val = threadIdx.x;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 1024) & (SMEM_BYTES/8 - 1);
            vsmem[s] = val + i + j;
        }
    }
}

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void sts128(uint4 *out, int N_iters) {
    __shared__ __align__(16) uint4 smem[SMEM_BYTES / 16];
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;  // each thread owns 1 uint4 (16 B), warp has 32 uint4
    unsigned int v0 = threadIdx.x, v1 = threadIdx.x+1, v2 = threadIdx.x+2, v3 = threadIdx.x+3;
    // Get shared base ptr as 32-bit offset
    unsigned smem_addr;
    smem_addr = (unsigned)__cvta_generic_to_shared(smem);
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 256) & (SMEM_BYTES/16 - 1);
            unsigned int byte_off = smem_addr + s * 16;
            unsigned int x = v0 + i + j, y = v1 + i + j, z = v2 + i + j, w = v3 + i + j;
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
                         :: "r"(byte_off), "r"(x), "r"(y), "r"(z), "r"(w));
        }
    }
}

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void lds32(int *out, int N_iters) {
    __shared__ int smem[SMEM_BYTES / 4];
    volatile int *vsmem = smem;
    for (int s = threadIdx.x; s < SMEM_BYTES/4; s += blockDim.x) smem[s] = s;
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
            int s = (slot + j * 1024 + i) & (SMEM_BYTES/4 - 1);
            acc ^= vsmem[s];
        }
    }
    if (acc == 0xDEADBEEF && N_iters < 0) out[threadIdx.x] = acc;
}

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void lds64(unsigned long long *out, int N_iters) {
    __shared__ unsigned long long smem[SMEM_BYTES / 8];
    volatile unsigned long long *vsmem = smem;
    for (int s = threadIdx.x; s < SMEM_BYTES/8; s += blockDim.x) smem[s] = s;
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    unsigned long long acc = 0;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 1024 + i) & (SMEM_BYTES/8 - 1);
            acc ^= vsmem[s];
        }
    }
    if (acc == 0xDEADBEEFULL && N_iters < 0) out[threadIdx.x] = acc;
}

template <int SMSP_MASK>
__launch_bounds__(1024, 1) __global__ void lds128(uint4 *out, int N_iters) {
    __shared__ __align__(16) uint4 smem[SMEM_BYTES / 16];
    for (int s = threadIdx.x; s < SMEM_BYTES/16; s += blockDim.x) {
        smem[s].x = s; smem[s].y = s+1; smem[s].z = s+2; smem[s].w = s+3;
    }
    __syncthreads();
    int warp_id = threadIdx.x >> 5;
    int smsp = warp_id & 3;
    if (((SMSP_MASK >> smsp) & 1) == 0) return;
    int lane = threadIdx.x & 31;
    int slot = warp_id * 32 + lane;
    unsigned smem_addr;
    smem_addr = (unsigned)__cvta_generic_to_shared(smem);
    unsigned int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int i = 0; i < N_iters; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            int s = (slot + j * 256 + i) & (SMEM_BYTES/16 - 1);
            unsigned int byte_off = smem_addr + s * 16;
            unsigned int x, y, z, w;
            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(byte_off));
            a0 ^= x; a1 ^= y; a2 ^= z; a3 ^= w;
        }
    }
    if ((a0 ^ a1 ^ a2 ^ a3) == 0xDEADBEEF && N_iters < 0) out[threadIdx.x].x = a0;
}

template <typename Fn, typename T>
void bench(const char* name, Fn kfn, T *d_out, int N_iters, int active_smsp, int bytes_per_op) {
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
    long active_threads = 8L * 32 * active_smsp * 148;
    long thr_ops = active_threads * (long)N_iters * N_INNER;
    long inst = thr_ops / 32;
    double clk_hz = 2032e6;
    double inst_per_sm_per_cy = (double)inst / (best/1000.0) / clk_hz / 148.0;
    double bytes_per_sec = (double)thr_ops * bytes_per_op / (best/1000.0);
    double bytes_per_sm_per_cy = bytes_per_sec / clk_hz / 148.0;
    printf("  %-25s  %.3f ms  %.3f inst/SM/cy  %6.2f TB/s  %5.1f B/SM/cy\n",
           name, best, inst_per_sm_per_cy, bytes_per_sec/1e12, bytes_per_sm_per_cy);
}

int main() {
    cudaSetDevice(0);
    int *d_out_i; cudaMalloc(&d_out_i, 1024 * sizeof(int));
    unsigned long long *d_out_u; cudaMalloc(&d_out_u, 1024 * sizeof(unsigned long long));
    uint4 *d_out_v; cudaMalloc(&d_out_v, 1024 * sizeof(uint4));
    int N = 200;
    printf("# === STS .32 ===\n");
    bench("STS.32 SMSP 0---", sts32<0b0001>, d_out_i, N, 1, 4);
    bench("STS.32 SMSPs 01--", sts32<0b0011>, d_out_i, N, 2, 4);
    bench("STS.32 SMSPs 0123", sts32<0b1111>, d_out_i, N, 4, 4);
    printf("# === STS .64 ===\n");
    bench("STS.64 SMSP 0---", sts64<0b0001>, d_out_u, N, 1, 8);
    bench("STS.64 SMSPs 01--", sts64<0b0011>, d_out_u, N, 2, 8);
    bench("STS.64 SMSPs 0123", sts64<0b1111>, d_out_u, N, 4, 8);
    printf("# === STS .128 ===\n");
    bench("STS.128 SMSP 0---", sts128<0b0001>, d_out_v, N, 1, 16);
    bench("STS.128 SMSPs 01--", sts128<0b0011>, d_out_v, N, 2, 16);
    bench("STS.128 SMSPs 0123", sts128<0b1111>, d_out_v, N, 4, 16);

    printf("\n# === LDS .32 ===\n");
    bench("LDS.32 SMSP 0---", lds32<0b0001>, d_out_i, N, 1, 4);
    bench("LDS.32 SMSPs 01--", lds32<0b0011>, d_out_i, N, 2, 4);
    bench("LDS.32 SMSPs 0123", lds32<0b1111>, d_out_i, N, 4, 4);
    printf("# === LDS .64 ===\n");
    bench("LDS.64 SMSP 0---", lds64<0b0001>, d_out_u, N, 1, 8);
    bench("LDS.64 SMSPs 01--", lds64<0b0011>, d_out_u, N, 2, 8);
    bench("LDS.64 SMSPs 0123", lds64<0b1111>, d_out_u, N, 4, 8);
    printf("# === LDS .128 ===\n");
    bench("LDS.128 SMSP 0---", lds128<0b0001>, d_out_v, N, 1, 16);
    bench("LDS.128 SMSPs 01--", lds128<0b0011>, d_out_v, N, 2, 16);
    bench("LDS.128 SMSPs 0123", lds128<0b1111>, d_out_v, N, 4, 16);
    return 0;
}
