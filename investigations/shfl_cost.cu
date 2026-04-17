// Measure __shfl costs on B300
#include <cuda_runtime.h>
#include <cstdio>

#define ITERS 10000

extern "C" __global__ void shfl_test(unsigned long long *out) {
    int tid = threadIdx.x;
    float val = (float)tid;
    unsigned long long start, end;

    // Test 1: __shfl_sync (full-mask)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        val = __shfl_sync(0xffffffff, val, (tid + 1) & 31);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[0] = end - start;

    // Test 2: __shfl_xor_sync
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        val = __shfl_xor_sync(0xffffffff, val, 1);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[1] = end - start;

    // Test 3: __shfl_down_sync
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        val = __shfl_down_sync(0xffffffff, val, 1);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[2] = end - start;

    // Test 4: __shfl_up_sync
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        val = __shfl_up_sync(0xffffffff, val, 1);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[3] = end - start;

    // Test 5: __ballot_sync
    int pred = (tid & 1);
    unsigned int ballot_result = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        ballot_result ^= __ballot_sync(0xffffffff, pred);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[4] = end - start;
        out[5] = ballot_result;  // DCE defense
    }

    // Test 6: __popc
    unsigned int popc_sum = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) {
        popc_sum += __popc(i * 31 + tid);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[6] = end - start;
        out[7] = popc_sum;
    }

    if (tid == 0 && val == -42.0f) out[8] = (unsigned long long)val;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out;
    cudaMalloc(&d_out, 16 * sizeof(unsigned long long));

    shfl_test<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();

    unsigned long long h[9];
    cudaMemcpy(h, d_out, 9 * 8, cudaMemcpyDeviceToHost);

    printf("# B300 warp-level primitive costs (%d iters)\n\n", ITERS);
    printf("  __shfl_sync:       %llu cy = %.2f cy/iter\n", h[0], h[0]/(double)ITERS);
    printf("  __shfl_xor_sync:   %llu cy = %.2f cy/iter\n", h[1], h[1]/(double)ITERS);
    printf("  __shfl_down_sync:  %llu cy = %.2f cy/iter\n", h[2], h[2]/(double)ITERS);
    printf("  __shfl_up_sync:    %llu cy = %.2f cy/iter\n", h[3], h[3]/(double)ITERS);
    printf("  __ballot_sync:     %llu cy = %.2f cy/iter\n", h[4], h[4]/(double)ITERS);
    printf("  __popc:            %llu cy = %.2f cy/iter\n", h[6], h[6]/(double)ITERS);

    cudaFree(d_out);
    return 0;
}
