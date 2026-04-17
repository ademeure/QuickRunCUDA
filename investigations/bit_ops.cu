// Bit manipulation throughput on B300
#include <cuda_runtime.h>
#include <cstdio>

#define ITERS 10000

extern "C" __global__ void bit_test(int *in, int *out) {
    int tid = threadIdx.x;
    int v = in[tid & 15];
    unsigned long long start, end;

    // __brev (bit reverse)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        v = __brev(v + i);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[0] = (int)(end - start);

    // __clz (count leading zeros)
    v = in[0];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        v = __clz(v + i + 1);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[1] = (int)(end - start);

    // __ffs (find first set)
    v = in[0];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        v = __ffs(v + i + 1);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[2] = (int)(end - start);

    // LOP3 (arbitrary 3-input bit op)
    v = in[0];
    int a = in[0], b = in[1], c = in[2];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v) : "r"(a), "r"(b));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[3] = (int)(end - start);
        out[4] = v;
    }

    // PRMT (byte permute)
    v = in[0];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        asm volatile("prmt.b32 %0, %0, %1, 0x1234;" : "+r"(v) : "r"(a));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[5] = (int)(end - start);
        out[6] = v;
    }

    // BFE (bit field extract)
    v = in[0];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        asm volatile("bfe.u32 %0, %0, 4, 8;" : "+r"(v));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[7] = (int)(end - start);
        out[8] = v;
    }
}

int main() {
    cudaSetDevice(0);
    int *d_in, *d_out;
    cudaMalloc(&d_in, 16 * sizeof(int));
    cudaMalloc(&d_out, 16 * sizeof(int));
    int h_in[16]; for (int i = 0; i < 16; i++) h_in[i] = 0x12345678 + i;
    cudaMemcpy(d_in, h_in, 16 * sizeof(int), cudaMemcpyHostToDevice);

    bit_test<<<1, 32>>>(d_in, d_out);
    cudaDeviceSynchronize();

    int h[16];
    cudaMemcpy(h, d_out, 16 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("# B300 bit manipulation op throughput (1 warp, %d iters)\n\n", ITERS);
    printf("  __brev (bit reverse) : %d cy = %.2f cy/iter\n", h[0], (double)h[0]/ITERS);
    printf("  __clz                : %d cy = %.2f cy/iter\n", h[1], (double)h[1]/ITERS);
    printf("  __ffs                : %d cy = %.2f cy/iter\n", h[2], (double)h[2]/ITERS);
    printf("  LOP3.b32             : %d cy = %.2f cy/iter\n", h[3], (double)h[3]/ITERS);
    printf("  PRMT.b32             : %d cy = %.2f cy/iter\n", h[5], (double)h[5]/ITERS);
    printf("  BFE.u32              : %d cy = %.2f cy/iter\n", h[7], (double)h[7]/ITERS);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
