// Quick verification of fence cost claims
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void test(int *out) {
    __shared__ int flag;
    if (threadIdx.x == 0) flag = 0;
    __syncthreads();

    unsigned long long start, end;

    // Fence.sc.cta
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < 100; i++) {
        asm volatile("fence.sc.cta;" ::: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (threadIdx.x == 0) out[0] = (int)(end - start);

    // Fence.sc.gpu
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < 100; i++) {
        asm volatile("fence.sc.gpu;" ::: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (threadIdx.x == 0) out[1] = (int)(end - start);

    // Fence.sc.sys
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < 100; i++) {
        asm volatile("fence.sc.sys;" ::: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (threadIdx.x == 0) out[2] = (int)(end - start);

    // Fence.acq_rel.cta
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < 100; i++) {
        asm volatile("fence.acq_rel.cta;" ::: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (threadIdx.x == 0) out[3] = (int)(end - start);

    // Fence.acq_rel.gpu
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < 100; i++) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (threadIdx.x == 0) out[4] = (int)(end - start);
}

int main() {
    cudaSetDevice(0);
    int *d_out;
    cudaMalloc(&d_out, 16 * sizeof(int));

    test<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();

    int h[5];
    cudaMemcpy(h, d_out, 5 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("# B300 fence cost verification (100 fences per kind)\n");
    printf("# Per-fence cost in cycles:\n");
    printf("  fence.sc.cta:      %.2f cy\n", h[0] / 100.0);
    printf("  fence.sc.gpu:      %.2f cy\n", h[1] / 100.0);
    printf("  fence.sc.sys:      %.2f cy\n", h[2] / 100.0);
    printf("  fence.acq_rel.cta: %.2f cy\n", h[3] / 100.0);
    printf("  fence.acq_rel.gpu: %.2f cy\n", h[4] / 100.0);

    cudaFree(d_out);
    return 0;
}
