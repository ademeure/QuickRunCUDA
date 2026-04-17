// Measure __syncthreads vs various sync primitives on B300
#include <cuda_runtime.h>
#include <cstdio>

#define ITERS 1000

extern "C" __global__ void sync_test(unsigned long long *out) {
    int tid = threadIdx.x;
    unsigned long long start, end;

    // Test 1: __syncthreads (classic)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        __syncthreads();
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[0] = end - start;

    // Test 2: __syncwarp
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        __syncwarp();
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[1] = end - start;

    // Test 3: bar.sync (PTX-level)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        asm volatile("bar.sync 0;" ::: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[2] = end - start;

    // Test 4: membar.cta
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        asm volatile("membar.cta;" ::: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[3] = end - start;

    // Test 5: membar.gl
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        asm volatile("membar.gl;" ::: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[4] = end - start;

    // Test 6: membar.sys
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        asm volatile("membar.sys;" ::: "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[5] = end - start;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out;
    cudaMalloc(&d_out, 16 * sizeof(unsigned long long));

    sync_test<<<1, 128>>>(d_out);
    cudaDeviceSynchronize();

    unsigned long long h[6];
    cudaMemcpy(h, d_out, 6 * 8, cudaMemcpyDeviceToHost);

    printf("# B300 sync primitive costs (1 block × 128 threads, %d iters)\n\n", ITERS);
    printf("  __syncthreads:  %llu cy = %.2f cy/iter\n", h[0], h[0]/(double)ITERS);
    printf("  __syncwarp:     %llu cy = %.2f cy/iter\n", h[1], h[1]/(double)ITERS);
    printf("  bar.sync 0:     %llu cy = %.2f cy/iter\n", h[2], h[2]/(double)ITERS);
    printf("  membar.cta:     %llu cy = %.2f cy/iter\n", h[3], h[3]/(double)ITERS);
    printf("  membar.gl:      %llu cy = %.2f cy/iter\n", h[4], h[4]/(double)ITERS);
    printf("  membar.sys:     %llu cy = %.2f cy/iter\n", h[5], h[5]/(double)ITERS);

    cudaFree(d_out);
    return 0;
}
