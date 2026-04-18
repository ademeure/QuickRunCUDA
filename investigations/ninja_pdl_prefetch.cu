// B5: PDL meaningful workload — HBM prefetch overlap with compute
//
// Pattern:
//   A: HMMA (compute-bound, no HBM access)
//   B: HBM read (memory-bound, no compute deps on A)
//
// Without PDL: t(A→B) ≈ t(A) + t(B)
// With PDL:    A launches B early via griddepcontrol.launch_dependents
//              B's HBM read overlaps with A's tail compute
//              t(A→B) ≈ max(t(A), t(B))
//
// Need cudaLaunchAttributeProgrammaticStreamSerialization on the stream.
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

// Kernel A: compute-only, signals dependents early
__launch_bounds__(512, 1) __global__ void k_compute(int *out, int N, bool use_pdl) {
    if (use_pdl) {
        // Allow dependent kernels to launch concurrently
        asm volatile("griddepcontrol.launch_dependents;\n" : : : "memory");
    }
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

// Kernel B: pure HBM read (independent of A's output), waits for A's signal
__launch_bounds__(256, 8) __global__ void k_hbm(const int4 *p, int *out, size_t N, int reps, bool use_pdl) {
    if (use_pdl) {
        // Wait for the parent (A) to finish (semantically; but B's prefetch already started)
        asm volatile("griddepcontrol.wait;\n" : : : "memory");
    }
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    int4 acc = make_int4(0,0,0,0);
    for (int r = 0; r < reps; r++) {
        for (size_t i = tid; i < N; i += stride) {
            int4 v = p[i];
            acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xDEADBEEF && reps < 0)
        out[threadIdx.x] = acc.x;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    size_t buf_bytes = 4ull * 1024 * 1024 * 1024;
    int4 *d_data; cudaMalloc(&d_data, buf_bytes);
    cudaMemset(d_data, 0, buf_bytes);
    size_t N_int4 = buf_bytes / 16;
    int N = 200;  // ~1.5 ms HMMA
    int reps = 1; // ~0.6 ms HBM read of 4 GB
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    // Standard stream (no PDL serialization attribute)
    cudaStream_t s_std; cudaStreamCreate(&s_std);
    // PDL stream
    cudaStream_t s_pdl; cudaStreamCreate(&s_pdl);

    auto bench_pair = [&](const char* name, cudaStream_t s, bool use_pdl, int chain_len) {
        // Setup PDL launch attribute on each launch
        cudaLaunchAttribute la = {};
        la.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        la.val.programmaticStreamSerializationAllowed = use_pdl ? 1 : 0;
        cudaLaunchConfig_t cfg_a = {};
        cfg_a.gridDim = dim3(148);
        cfg_a.blockDim = dim3(512);
        cfg_a.stream = s;
        cfg_a.attrs = &la;
        cfg_a.numAttrs = 1;
        cudaLaunchConfig_t cfg_b = {};
        cfg_b.gridDim = dim3(148*8);
        cfg_b.blockDim = dim3(256);
        cfg_b.stream = s;
        cfg_b.attrs = &la;
        cfg_b.numAttrs = 1;

        // Warmup
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < chain_len; j++) {
                cudaLaunchKernelEx(&cfg_a, k_compute, d_out, N, use_pdl);
                cudaLaunchKernelEx(&cfg_b, k_hbm, (const int4*)d_data, d_out, N_int4, reps, use_pdl);
            }
        }
        cudaStreamSynchronize(s);

        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0, s);
            for (int j = 0; j < chain_len; j++) {
                cudaLaunchKernelEx(&cfg_a, k_compute, d_out, N, use_pdl);
                cudaLaunchKernelEx(&cfg_b, k_hbm, (const int4*)d_data, d_out, N_int4, reps, use_pdl);
            }
            cudaEventRecord(e1, s); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double per_pair = best / chain_len;
        printf("  %-25s chain=%d  total=%.2f ms  per pair=%.3f ms\n",
               name, chain_len, best, per_pair);
        return per_pair;
    };

    // Time A alone
    {
        cudaLaunchAttribute la = {};
        la.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        la.val.programmaticStreamSerializationAllowed = 0;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(148);
        cfg.blockDim = dim3(512);
        cfg.stream = s_std;
        cfg.attrs = &la;
        cfg.numAttrs = 1;
        for (int i = 0; i < 3; i++) cudaLaunchKernelEx(&cfg, k_compute, d_out, N, false);
        cudaStreamSynchronize(s_std);
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0, s_std);
            cudaLaunchKernelEx(&cfg, k_compute, d_out, N, false);
            cudaEventRecord(e1, s_std); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        printf("# A (k_compute) alone: %.3f ms\n", best);
    }
    {
        cudaLaunchAttribute la = {};
        la.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        la.val.programmaticStreamSerializationAllowed = 0;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(148*8);
        cfg.blockDim = dim3(256);
        cfg.stream = s_std;
        cfg.attrs = &la;
        cfg.numAttrs = 1;
        for (int i = 0; i < 3; i++) cudaLaunchKernelEx(&cfg, k_hbm, (const int4*)d_data, d_out, N_int4, reps, false);
        cudaStreamSynchronize(s_std);
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0, s_std);
            cudaLaunchKernelEx(&cfg, k_hbm, (const int4*)d_data, d_out, N_int4, reps, false);
            cudaEventRecord(e1, s_std); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        printf("# B (k_hbm) alone:     %.3f ms\n", best);
    }

    printf("# Chain A→B (back-to-back):\n");
    double t_no  = bench_pair("Standard (no PDL)", s_std, false, 8);
    double t_pdl = bench_pair("PDL (overlap)",     s_pdl, true,  8);
    printf("# PDL savings: %.3f ms/pair = %.1f%%\n", t_no - t_pdl, (t_no - t_pdl)/t_no*100);

    return 0;
}
