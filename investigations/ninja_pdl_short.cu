// Test PDL with SHORT kernels (where launch overhead matters)
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(64, 16) __global__ void k_a(int *out, int N, bool use_pdl) {
    if (use_pdl) asm volatile("griddepcontrol.launch_dependents;\n" : : : "memory");
    float a = threadIdx.x * 0.1f, b = 0.5f, c = 0.001f;
    for (int i = 0; i < N; i++) a = a * b + c;
    if (a == 0xDEADBEEF && N < 0) out[threadIdx.x] = (int)a;
}
__launch_bounds__(64, 16) __global__ void k_b(int *out, int N, bool use_pdl) {
    if (use_pdl) asm volatile("griddepcontrol.wait;\n" : : : "memory");
    float a = threadIdx.x * 0.1f, b = 0.5f, c = 0.001f;
    for (int i = 0; i < N; i++) a = a * b + c;
    if (a == 0xDEADBEEF && N < 0) out[threadIdx.x] = (int)a;
}
int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto run = [&](const char* name, bool pdl, int N, int chain) {
        cudaLaunchAttribute la = {};
        la.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        la.val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(148);
        cfg.blockDim = dim3(64);
        cfg.stream = s;
        cfg.attrs = &la;
        cfg.numAttrs = 1;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < chain; j++) {
                cudaLaunchKernelEx(&cfg, k_a, d_out, N, pdl);
                cudaLaunchKernelEx(&cfg, k_b, d_out, N, pdl);
            }
        }
        cudaStreamSynchronize(s);
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0, s);
            for (int j = 0; j < chain; j++) {
                cudaLaunchKernelEx(&cfg, k_a, d_out, N, pdl);
                cudaLaunchKernelEx(&cfg, k_b, d_out, N, pdl);
            }
            cudaEventRecord(e1, s); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        printf("  %-20s N=%-6d chain=%3d  total=%.3f ms  per pair=%.2f us\n",
               name, N, chain, best, best/chain*1000);
    };
    printf("# Short kernel chains (FFMA only, 64 thr/blk, varying N)\n");
    for (int N : {100, 1000, 10000}) {
        run("standard", false, N, 64);
        run("PDL",      true,  N, 64);
    }
    return 0;
}
