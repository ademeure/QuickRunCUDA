// S1: mma.sync warp-count saturation
//
// CURIOSITY_LIST claim: "mma.sync caps at 570 TFLOPS = 23% of NVIDIA's 2.5 PF spec"
// MY recent HMMA test (raw mma.sync.aligned.m16n8k16, 32 warps, 4 chains): 2.21 PFLOPS = 88% of spec
//
// Either curiosity claim wrong, or my measurement bogus. To find out, sweep warps/block.
//
// THEORETICAL:
//   B300 BF16 dense via mma.sync, m16n8k16: 2 × 16 × 8 × 16 = 4096 FLOPS per inst
//   NVIDIA spec: 2500 TFLOPS BF16 dense
//   At 2032 MHz: 2500e12 / 2.032e9 / 148 = 8311 FLOPS/SM/cy = 2.03 mma/SM/cy needed
//
// TEST: vary block size = warps/SM from 1, 2, 4, 8, 16, 32. Measure rate.
//   - If 1 warp gets 1/32 of peak: per-SM saturation needs 32 warps
//   - If 4 warps get 4/32 = 1/8 of peak: per-SMSP saturation needs 32 warps (8/SMSP)
//   - If 4 warps get FULL peak: per-SMSP cap = 1 warp/SMSP

#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 64;
constexpr int N_CHAINS = 4;

__device__ __forceinline__ void mma_b16(unsigned (&d)[4],
                                        unsigned (&a)[4], unsigned (&b)[2],
                                        unsigned (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

template <int THREADS>
__launch_bounds__(THREADS, 1) __global__ void k_hmma(int *out, int N) {
    unsigned a0[4] = {0x3f800001, 0x3f800002, 0x3f800003, 0x3f800004};
    unsigned a1[4] = {0x3f800005, 0x3f800006, 0x3f800007, 0x3f800008};
    unsigned a2[4] = {0x3f800009, 0x3f80000a, 0x3f80000b, 0x3f80000c};
    unsigned a3[4] = {0x3f80000d, 0x3f80000e, 0x3f80000f, 0x3f800010};
    unsigned b0[2] = {0x3f800001, 0x3f800002};
    unsigned b1[2] = {0x3f800003, 0x3f800004};
    unsigned b2[2] = {0x3f800005, 0x3f800006};
    unsigned b3[2] = {0x3f800007, 0x3f800008};
    unsigned c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_b16(c0, a0, b0, c0);
            mma_b16(c1, a1, b1, c1);
            mma_b16(c2, a2, b2, c2);
            mma_b16(c3, a3, b3, c3);
        }
    }
    if (c0[0] == 0xDEADBEEFu && N < 0) out[threadIdx.x] = c0[0];
}

template <int THREADS>
double bench(const char* name, int *d_out, int N) {
    int blocks = 148;  // 1 block per SM
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) k_hmma<THREADS><<<blocks, THREADS>>>(d_out, N);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s: %s\n", name, cudaGetErrorString(cudaGetLastError())); return 0; }
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        k_hmma<THREADS><<<blocks, THREADS>>>(d_out, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    int warps_per_block = THREADS / 32;
    long mma_per_warp = (long)N * N_INNER * N_CHAINS;
    long total_mma = (long)blocks * warps_per_block * mma_per_warp;
    double pflops = (double)total_mma * 4096.0 / (best/1000.0) / 1e15;
    double mma_per_sm_per_cy = (double)total_mma / 148.0 / (best/1000.0) / 2.032e9;
    double pct_spec = pflops / 2.5 * 100.0;
    printf("  %-25s warps/SM=%2d  %.3f ms  %.2f PFLOPS  %.2f mma/SM/cy  (%.1f%% of 2.5 PF spec)\n",
           name, warps_per_block, best, pflops, mma_per_sm_per_cy, pct_spec);
    return pflops;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    printf("# mma.sync.m16n8k16 BF16 dense — warp-count sweep, 4 chains/warp\n\n");
    bench<32>(  "1 warp/SM   (1 SMSP)",   d_out, 800);
    bench<64>(  "2 warps/SM  (2 SMSPs)",  d_out, 600);
    bench<128>( "4 warps/SM  (4 SMSPs, 1 each)", d_out, 400);
    bench<256>( "8 warps/SM  (4 SMSPs, 2 each)", d_out, 300);
    bench<512>( "16 warps/SM (4 SMSPs, 4 each)", d_out, 200);
    bench<1024>("32 warps/SM (4 SMSPs, 8 each)", d_out, 200);
    return 0;
}
