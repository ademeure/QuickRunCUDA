// Check fence cost impact of pending uncoalesced vs coalesced stores
#include <cuda_runtime.h>
#include <cstdio>

#define N_STORES 32
#define N_FENCES 10

extern "C" __global__ void coalesced(int *out, int *buf) {
    int tid = threadIdx.x;

    unsigned long long start;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    // COALESCED: thread N writes to buf[N]
    for (int i = 0; i < N_STORES; i++) {
        buf[i * blockDim.x + tid] = tid + i;
    }

    // Now fence
    for (int f = 0; f < N_FENCES; f++) {
        asm volatile("fence.sc.gpu;" ::: "memory");
    }

    unsigned long long end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));

    if (tid == 0) out[0] = (int)(end - start);
}

extern "C" __global__ void uncoalesced(int *out, int *buf) {
    int tid = threadIdx.x;

    unsigned long long start;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    // UNCOALESCED: thread N writes to buf[N * stride]
    // Each warp's 32 stores hit 32 different cache lines
    int stride = 32;  // each store to a different 128B cache line
    for (int i = 0; i < N_STORES; i++) {
        buf[i * blockDim.x * stride + tid * stride] = tid + i;
    }

    // Now fence
    for (int f = 0; f < N_FENCES; f++) {
        asm volatile("fence.sc.gpu;" ::: "memory");
    }

    unsigned long long end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));

    if (tid == 0) out[0] = (int)(end - start);
}

extern "C" __global__ void fence_only(int *out) {
    int tid = threadIdx.x;

    unsigned long long start;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));

    for (int f = 0; f < N_FENCES; f++) {
        asm volatile("fence.sc.gpu;" ::: "memory");
    }

    unsigned long long end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));

    if (tid == 0) out[0] = (int)(end - start);
}

int main() {
    cudaSetDevice(0);

    int *d_out;
    int *d_buf;
    cudaMalloc(&d_out, 16 * sizeof(int));
    cudaMalloc(&d_buf, 32 * 128 * 32 * sizeof(int));  // 4 MB buffer

    // Warmup
    for (int i = 0; i < 3; i++) {
        coalesced<<<1, 128>>>(d_out, d_buf);
        uncoalesced<<<1, 128>>>(d_out, d_buf);
        fence_only<<<1, 128>>>(d_out);
        cudaDeviceSynchronize();
    }

    int h[3];

    coalesced<<<1, 128>>>(d_out, d_buf);
    cudaDeviceSynchronize();
    cudaMemcpy(&h[0], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    uncoalesced<<<1, 128>>>(d_out, d_buf);
    cudaDeviceSynchronize();
    cudaMemcpy(&h[1], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    fence_only<<<1, 128>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(&h[2], d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("# B300 fence cost with pending stores\n");
    printf("# 128 threads, %d stores per thread, then %d fence.sc.gpu\n\n", N_STORES, N_FENCES);
    printf("  fence only (no stores):                 %d cy (%.1f per fence)\n",
           h[2], h[2] / (float)N_FENCES);
    printf("  coalesced stores + fences:              %d cy\n", h[0]);
    printf("  uncoalesced stores + fences:            %d cy\n", h[1]);
    printf("\n  Fence overhead (coalesced minus fence-only):   %d cy\n", h[0] - h[2]);
    printf("  Fence overhead (uncoalesced minus fence-only): %d cy\n", h[1] - h[2]);
    printf("  Ratio: %.2fx\n", (h[1] - h[2]) / (float)(h[0] - h[2]));

    cudaFree(d_out); cudaFree(d_buf);
    return 0;
}
