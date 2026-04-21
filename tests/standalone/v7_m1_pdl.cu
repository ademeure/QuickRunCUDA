// V7 M1: PDL (Programmatic Dependent Launch) kernel-to-kernel coordination
// Compare:
//   MODE 0: Two kernels in same stream (default sync at end of A → start of B)
//   MODE 1: PDL — B can start as A nears completion
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

__global__ void kernel_A(unsigned int* buf, int delay_iters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < (unsigned long long)delay_iters);
        // Trigger PDL — allow B to launch
        asm volatile("griddepcontrol.launch_dependents;");
        buf[0] = 1;
    }
}

__global__ void kernel_B(unsigned int* buf, int delay_iters) {
    // Wait for A's launch_dependents (only meaningful with PDL)
    asm volatile("griddepcontrol.wait;");
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < (unsigned long long)delay_iters);
        buf[1] = 2;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 16);

    int delay = 1500000;  // ~1 ms each kernel
    int N = 100;

    cudaStream_t s;
    cudaStreamCreate(&s);

    cudaEvent_t es, ee;
    cudaEventCreate(&es); cudaEventCreate(&ee);

    // Warmup
    kernel_A<<<1, 32, 0, s>>>(buf, delay);
    kernel_B<<<1, 32, 0, s>>>(buf, delay);
    cudaStreamSynchronize(s);

    // Test 1: regular launches (no PDL)
    cudaEventRecord(es, s);
    for (int i = 0; i < N; i++) {
        kernel_A<<<1, 32, 0, s>>>(buf, delay);
        kernel_B<<<1, 32, 0, s>>>(buf, delay);
    }
    cudaEventRecord(ee, s);
    cudaEventSynchronize(ee);
    float reg_ms;
    cudaEventElapsedTime(&reg_ms, es, ee);

    // Test 2: launch B with PDL attribute (cudaLaunchAttributeProgrammaticStreamSerialization)
    cudaLaunchConfig_t cfgB = {};
    cfgB.gridDim = dim3(1); cfgB.blockDim = dim3(32);
    cfgB.stream = s;
    cudaLaunchAttribute pdl_attr;
    pdl_attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attr.val.programmaticStreamSerializationAllowed = 1;
    cfgB.numAttrs = 1;
    cfgB.attrs = &pdl_attr;

    cudaEventRecord(es, s);
    for (int i = 0; i < N; i++) {
        kernel_A<<<1, 32, 0, s>>>(buf, delay);
        cudaLaunchKernelEx(&cfgB, kernel_B, buf, delay);
    }
    cudaEventRecord(ee, s);
    cudaEventSynchronize(ee);
    float pdl_ms;
    cudaEventElapsedTime(&pdl_ms, es, ee);

    printf("Regular A→B chain (no PDL): %.2f ms total = %.3f ms/pair\n", reg_ms, reg_ms / N);
    printf("PDL A→B chain:              %.2f ms total = %.3f ms/pair\n", pdl_ms, pdl_ms / N);
    printf("PDL speedup: %.2fx\n", reg_ms / pdl_ms);

    return 0;
}
