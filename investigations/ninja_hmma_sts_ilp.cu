// Why does HMMA+STS show ~40% overlap (not full independence)?
// Hypothesis: HMMA needs RF reads for A/B operands, contends with STS RF read.
// Test: vary HMMA chains (ILP) and STS frequency.
#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 64;

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

// HMMA-only with N chains
template<int N_CHAINS>
__launch_bounds__(512, 1) __global__ void k_hmma(int *out, int N) {
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
            if (N_CHAINS >= 1) mma_b16(c0, a0, b0, c0);
            if (N_CHAINS >= 2) mma_b16(c1, a1, b1, c1);
            if (N_CHAINS >= 3) mma_b16(c2, a2, b2, c2);
            if (N_CHAINS >= 4) mma_b16(c3, a3, b3, c3);
        }
    }
    if (c0[0] == 0xDEADBEEFu && N < 0) out[threadIdx.x] = c0[0];
}

// HMMA + STS — N HMMA chains, M STS per inner
template<int N_CHAINS, int M_STS>
__launch_bounds__(512, 1) __global__ void k_hmma_sts(int *out, int N) {
    __shared__ int smem[1024 * 4];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x >> 5) * 32 + (threadIdx.x & 31);
    int v = threadIdx.x;
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
            if (N_CHAINS >= 1) mma_b16(c0, a0, b0, c0);
            if (M_STS >= 1) vsmem[slot + 0*1024] = v + i;
            if (N_CHAINS >= 2) mma_b16(c1, a1, b1, c1);
            if (M_STS >= 2) vsmem[slot + 1*1024] = v + i;
            if (N_CHAINS >= 3) mma_b16(c2, a2, b2, c2);
            if (M_STS >= 3) vsmem[slot + 2*1024] = v + i;
            if (N_CHAINS >= 4) mma_b16(c3, a3, b3, c3);
            if (M_STS >= 4) vsmem[slot + 3*1024] = v + i;
        }
    }
    if (c0[0] == 0xDEADBEEFu && N < 0) out[threadIdx.x] = c0[0];
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    int N = 200;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto bench = [&](const char* name, auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(cudaGetLastError())); return 0.0f; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        printf("  %-25s  %.3f ms\n", name, best);
        return best;
    };
    printf("# HMMA+STS contention vs ILP\n# Measure: t(combo) vs t(HMMA alone) — if equal, free.\n\n");

    // HMMA alone with various chain counts
    float t_h1 = bench("HMMA chains=1",  [&]() { k_hmma<1><<<148, 512>>>(d_out, N); });
    float t_h2 = bench("HMMA chains=2",  [&]() { k_hmma<2><<<148, 512>>>(d_out, N); });
    float t_h4 = bench("HMMA chains=4",  [&]() { k_hmma<4><<<148, 512>>>(d_out, N); });

    // STS alone (no HMMA)
    float t_s1 = bench("STS only count=1", [&]() { k_hmma_sts<0, 1><<<148, 512>>>(d_out, N); });
    float t_s4 = bench("STS only count=4", [&]() { k_hmma_sts<0, 4><<<148, 512>>>(d_out, N); });

    printf("\n# Combos: (chains × STS count per inner)\n");
    bench("HMMA=1 + STS=1", [&]() { k_hmma_sts<1, 1><<<148, 512>>>(d_out, N); });
    bench("HMMA=2 + STS=2", [&]() { k_hmma_sts<2, 2><<<148, 512>>>(d_out, N); });
    bench("HMMA=4 + STS=4", [&]() { k_hmma_sts<4, 4><<<148, 512>>>(d_out, N); });
    bench("HMMA=4 + STS=1", [&]() { k_hmma_sts<4, 1><<<148, 512>>>(d_out, N); });
    bench("HMMA=1 + STS=4", [&]() { k_hmma_sts<1, 4><<<148, 512>>>(d_out, N); });
    return 0;
}
