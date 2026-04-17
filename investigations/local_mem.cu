// Local memory (stack spill) cost on B300
#include <cuda_runtime.h>
#include <cstdio>

#define ITERS 10000
#define N_REG 256  // Force spill: 256 floats = 1 KB per thread

extern "C" __global__ void local_mem_test(int *out, int start_idx) {
    // Force local mem via array indexed by runtime value (compiler can't register-alloc)
    float arr[N_REG];
    int tid = threadIdx.x;
    for (int i = 0; i < N_REG; i++) arr[i] = (float)(tid + i);

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    float acc = 0;
    for (int i = 0; i < ITERS; i++) {
        int idx = (start_idx + i * 7) & (N_REG - 1);  // runtime-computed index
        acc += arr[idx];
        arr[idx] = acc;  // Write too
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[0] = (int)acc;
        out[1] = (int)(end - start);
    }
}

int main() {
    cudaSetDevice(0);
    int *d_out;
    cudaMalloc(&d_out, 4 * sizeof(int));

    local_mem_test<<<1, 32>>>(d_out, 3);
    cudaDeviceSynchronize();
    cudaError_t r = cudaGetLastError();
    printf("Launch: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

    int h[2];
    cudaMemcpy(h, d_out, 2*sizeof(int), cudaMemcpyDeviceToHost);

    // Check function attributes for local memory
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, (void*)local_mem_test);
    printf("# B300 local memory cost test\n");
    printf("# Kernel uses %zu bytes local memory per thread, %d registers\n",
           fa.localSizeBytes, fa.numRegs);
    printf("# Single warp, %d iters of read+write to runtime-indexed array\n", ITERS);
    printf("  Total cy: %d\n", h[1]);
    printf("  Cy/iter:  %.2f (1 read + 1 write per iter = 2 LMEM ops)\n", (double)h[1]/ITERS);
    printf("  Cy/LMEM op: %.2f\n", (double)h[1]/ITERS/2);

    cudaFree(d_out);
    return 0;
}
