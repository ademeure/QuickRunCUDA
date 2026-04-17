// Compare cooperative_groups primitive costs vs raw equivalents
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

#define ITERS 1000

extern "C" __global__ void cg_costs(unsigned long long *out) {
    int tid = threadIdx.x;
    unsigned long long start, end;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    // Test 1: __syncthreads
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) __syncthreads();
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[0] = end - start;

    // Test 2: block.sync()
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) block.sync();
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[1] = end - start;

    // Test 3: warp.sync()
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) warp.sync();
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[2] = end - start;

    // Test 4: __syncwarp
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) __syncwarp();
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[3] = end - start;

    // Test 5: warp.shfl()
    int v = tid;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) v = warp.shfl(v, (tid + 1) & 31);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[4] = end - start;
        out[5] = v;  // DCE defense
    }

    // Test 6: __shfl_sync
    int u = tid;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) u = __shfl_sync(0xffffffff, u, (tid + 1) & 31);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[6] = end - start;
        out[7] = u;
    }

    // Test 7: warp.ballot
    int p = (tid & 1);
    unsigned int b = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < ITERS; i++) b ^= warp.ballot(p);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[8] = end - start;
        out[9] = b;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out;
    cudaMalloc(&d_out, 16 * sizeof(unsigned long long));

    cg_costs<<<1, 128>>>(d_out);
    cudaDeviceSynchronize();

    unsigned long long h[10];
    cudaMemcpy(h, d_out, 10 * 8, cudaMemcpyDeviceToHost);

    printf("# B300 cooperative_groups vs raw primitives (1 block × 128 threads, %d iters)\n\n", ITERS);
    printf("  __syncthreads()              : %.2f cy/iter\n", h[0]/(double)ITERS);
    printf("  block.sync()                 : %.2f cy/iter\n", h[1]/(double)ITERS);
    printf("  __syncwarp()                 : %.2f cy/iter\n", h[3]/(double)ITERS);
    printf("  warp.sync()                  : %.2f cy/iter\n", h[2]/(double)ITERS);
    printf("  __shfl_sync()                : %.2f cy/iter\n", h[6]/(double)ITERS);
    printf("  warp.shfl()                  : %.2f cy/iter\n", h[4]/(double)ITERS);
    printf("  warp.ballot() = ballot_sync  : %.2f cy/iter\n", h[8]/(double)ITERS);

    cudaFree(d_out);
    return 0;
}
