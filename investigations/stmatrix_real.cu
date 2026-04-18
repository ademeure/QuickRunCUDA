// B3 RIGOR v2: stmatrix with HARD data dependency that defeats DCE.
// stmatrix writes -> __syncwarp -> read EXACTLY the bytes stmatrix wrote -> sum
// Re-feed sum into next iter's stmatrix payload so each iter depends on prior.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

extern "C" __launch_bounds__(128, 4) __global__ void stmatrix_chain(
    __half *gin, float *gout, int iters)
{
    extern __shared__ __half smem[];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    __half *wsmem = smem + warp * 1024;  // 2048 B per warp

    // Initial values from gmem to vary per warp
    int g_off = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned r0 = ((unsigned*)gin)[g_off * 4 + 0];
    unsigned r1 = ((unsigned*)gin)[g_off * 4 + 1];
    unsigned r2 = ((unsigned*)gin)[g_off * 4 + 2];
    unsigned r3 = ((unsigned*)gin)[g_off * 4 + 3];

    // Each lane in the warp writes to row `lane` of the matrix.
    // stmatrix x4 has 4 matrices, each 8x8 = 256 halves total per matrix
    // -> 4 matrices = 1024 halves total = 2048 B per warp
    // -> per lane contributes 8 halves = 16 B (4 unsigned values)
    unsigned smem_addr;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(smem_addr) : "l"(wsmem + lane * 8));

    float acc = 0.0f;

    #pragma unroll 1
    for (int it = 0; it < iters; it++) {
        // stmatrix writes EXACTLY 16 B at addr (smem_addr) for this lane
        asm volatile(
            "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
            :: "r"(smem_addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
        __syncwarp();

        // Read back the bytes THIS lane just wrote
        unsigned v0, v1, v2, v3;
        asm volatile("ld.shared.b32 %0, [%1];"     : "=r"(v0) : "r"(smem_addr));
        asm volatile("ld.shared.b32 %0, [%1+4];"   : "=r"(v1) : "r"(smem_addr));
        asm volatile("ld.shared.b32 %0, [%1+8];"   : "=r"(v2) : "r"(smem_addr));
        asm volatile("ld.shared.b32 %0, [%1+12];"  : "=r"(v3) : "r"(smem_addr));
        // Mix into accumulator AND mutate next-iter regs
        unsigned x = v0 ^ v1 ^ v2 ^ v3;
        acc += __half2float(*(__half*)&x);
        // Mutate so iters chain
        r0 = v0 + 1;
        r1 = v1 + 1;
        r2 = v2 + 1;
        r3 = v3 + 1;
    }

    if (acc > -1e30f) gout[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    int blocks = 148 * 4, threads = 128;
    int iters = (argc > 1) ? atoi(argv[1]) : 256;

    int n_in = blocks * threads * 8;
    __half *d_in; cudaMalloc(&d_in, n_in * sizeof(__half));
    cudaMemset(d_in, 0x12, n_in * sizeof(__half));
    float *d_out; cudaMalloc(&d_out, blocks * threads * sizeof(float));
    int smem_bytes = 1024 * sizeof(__half) * (threads / 32);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int i = 0; i < 5; i++)
        stmatrix_chain<<<blocks, threads, smem_bytes>>>(d_in, d_out, iters);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        stmatrix_chain<<<blocks, threads, smem_bytes>>>(d_in, d_out, iters);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    long warps = blocks * (threads / 32);
    long total_bytes = warps * (long)iters * 512;  // 4 mat × 8x8 × 2 B = 512 B/warp/call
    double tbs = total_bytes / (best/1000) / 1e12;

    printf("# stmatrix_chain v2: blocks=%d threads=%d iters=%d\n", blocks, threads, iters);
    printf("  Time: %.4f ms\n", best);
    printf("  Total stmatrix bytes: %.2f GB\n", total_bytes / 1e9);
    printf("  Effective SHMEM write BW: %.2f TB/s (= %.2f%% of 38.5 TB/s SoL)\n",
           tbs, tbs / 38.5 * 100);
    printf("  Per-SM: %.2f GB/s\n", tbs * 1000 / 148);

    return 0;
}
