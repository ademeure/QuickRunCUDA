// B3 RIGOR: stmatrix-HEAVY variant — many stmatrix per readback for pure write peak
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

#ifndef STM_PER_LDS
#define STM_PER_LDS 8
#endif

extern "C" __launch_bounds__(128, 4) __global__ void stmatrix_heavy(
    __half *gin, float *gout, int iters)
{
    extern __shared__ __half smem[];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    __half *wsmem = smem + warp * 1024;

    int g_off = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned r0 = ((unsigned*)gin)[g_off * 4 + 0];
    unsigned r1 = ((unsigned*)gin)[g_off * 4 + 1];
    unsigned r2 = ((unsigned*)gin)[g_off * 4 + 2];
    unsigned r3 = ((unsigned*)gin)[g_off * 4 + 3];

    unsigned smem_addr;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }"
        : "=r"(smem_addr) : "l"(wsmem + lane * 8));

    float acc = 0.0f;

    #pragma unroll 1
    for (int it = 0; it < iters; it++) {
        // Issue STM_PER_LDS stmatrix in tight sequence (each writes to different offset)
        #pragma unroll
        for (int s = 0; s < STM_PER_LDS; s++) {
            // Use a different sub-offset per s so they don't WAW-stall the same address
            unsigned addr = smem_addr;  // same addr — successive writes to same lines
            asm volatile(
                "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
                :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
            r0 ^= s;
            r1 ^= s;
        }
        __syncwarp();
        // ONE readback to force ALL stmatrix writes to retire
        unsigned v0, v1, v2, v3;
        asm volatile("ld.shared.b32 %0, [%1];"     : "=r"(v0) : "r"(smem_addr));
        asm volatile("ld.shared.b32 %0, [%1+4];"   : "=r"(v1) : "r"(smem_addr));
        asm volatile("ld.shared.b32 %0, [%1+8];"   : "=r"(v2) : "r"(smem_addr));
        asm volatile("ld.shared.b32 %0, [%1+12];"  : "=r"(v3) : "r"(smem_addr));
        unsigned x = v0 ^ v1 ^ v2 ^ v3;
        acc += __half2float(*(__half*)&x);
        r0 = v0 + 1;  r1 = v1 + 1;  r2 = v2 + 1;  r3 = v3 + 1;
    }

    if (acc > -1e30f) gout[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    int blocks = 148 * 4, threads = 128;
    int iters = (argc > 1) ? atoi(argv[1]) : 1024;

    int n_in = blocks * threads * 8;
    __half *d_in; cudaMalloc(&d_in, n_in * sizeof(__half));
    cudaMemset(d_in, 0x12, n_in * sizeof(__half));
    float *d_out; cudaMalloc(&d_out, blocks * threads * sizeof(float));
    int smem_bytes = 1024 * sizeof(__half) * (threads / 32);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int i = 0; i < 5; i++)
        stmatrix_heavy<<<blocks, threads, smem_bytes>>>(d_in, d_out, iters);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        stmatrix_heavy<<<blocks, threads, smem_bytes>>>(d_in, d_out, iters);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    long warps = blocks * (threads / 32);
    long stm_per_iter = STM_PER_LDS;
    long total_write_bytes = warps * (long)iters * stm_per_iter * 512;
    long total_read_bytes = warps * (long)iters * 512;  // 32 lanes × 16 B
    long total_bytes = total_write_bytes + total_read_bytes;

    double tbs_w = total_write_bytes / (best/1000) / 1e12;
    double tbs_total = total_bytes / (best/1000) / 1e12;

    printf("# stmatrix_heavy (STM_PER_LDS=%d, iters=%d): %.4f ms\n", STM_PER_LDS, iters, best);
    printf("  Write only: %.2f TB/s (%.1f%% of 38.5 SoL)\n", tbs_w, tbs_w/38.5*100);
    printf("  W+R total : %.2f TB/s (%.1f%% of 38.5 SoL)\n", tbs_total, tbs_total/38.5*100);
    printf("  Per-SM W only: %.1f GB/s\n", tbs_w*1000/148);

    return 0;
}
