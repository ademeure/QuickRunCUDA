// Test ldmatrix throughput on B300
#include <cuda_runtime.h>
#include <cstdio>

#define ITERS 10000

extern "C" __global__ void ldmatrix_test(unsigned long long *out) {
    __shared__ __align__(16) float smem[1024];
    int tid = threadIdx.x;
    smem[tid] = (float)tid;
    if (tid < 32) smem[32 + tid] = (float)(tid + 100);
    __syncthreads();

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    // ldmatrix.x4 - loads 4 matrix fragments (8x8 each) into 4 registers per thread
    unsigned int r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    unsigned int smem_addr = __cvta_generic_to_shared(&smem[0]);

    for (int i = 0; i < ITERS; i++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                     : "r"(smem_addr));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[0] = end - start;
        out[1] = r0 ^ r1 ^ r2 ^ r3;  // DCE defense
    }
}

extern "C" __global__ void ldmatrix_x1_test(unsigned long long *out) {
    __shared__ __align__(16) float smem[1024];
    int tid = threadIdx.x;
    smem[tid] = (float)tid;
    __syncthreads();

    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    unsigned int r0 = 0;
    unsigned int smem_addr = __cvta_generic_to_shared(&smem[0]);

    for (int i = 0; i < ITERS; i++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                     : "=r"(r0) : "r"(smem_addr));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[0] = end - start;
        out[1] = r0;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out;
    cudaMalloc(&d_out, 8 * sizeof(unsigned long long));

    ldmatrix_test<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    unsigned long long h_x4[2];
    cudaMemcpy(h_x4, d_out, 16, cudaMemcpyDeviceToHost);

    ldmatrix_x1_test<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    unsigned long long h_x1[2];
    cudaMemcpy(h_x1, d_out, 16, cudaMemcpyDeviceToHost);

    printf("# B300 ldmatrix throughput (single warp, %d iters)\n\n", ITERS);
    printf("  ldmatrix.m8n8.x4: %llu cy = %.2f cy/iter (128 bytes/iter = %.1f bytes/cy)\n",
           h_x4[0], (double)h_x4[0]/ITERS, 128.0 / ((double)h_x4[0]/ITERS));
    printf("  ldmatrix.m8n8.x1: %llu cy = %.2f cy/iter (32 bytes/iter = %.1f bytes/cy)\n",
           h_x1[0], (double)h_x1[0]/ITERS, 32.0 / ((double)h_x1[0]/ITERS));

    cudaFree(d_out);
    return 0;
}
