// 2-GPU all-reduce via NVLink P2P
// Theoretical SoL: each GPU reads the OTHER's buffer once = 1 buffer's worth
// of NVLink traffic. At 783 GB/s peak read NVLink: B/783e9 seconds.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

// Each GPU reads peer's buffer, sums into local, writes result to local
// At end, both GPUs have the sum (= 2x avg).
__launch_bounds__(256, 8) __global__ void allreduce_pair(
    __nv_bfloat16 *local, const __nv_bfloat16 *peer, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    // Each warp handles 8 BF16 per thread × 32 lanes = 256 BF16 = 512 B per warp
    int idx = (warp_id * 32 + lane) * 8;
    if (idx + 7 >= N) return;
    uint4 v_local = *(uint4*)&local[idx];
    uint4 v_peer = *(uint4*)&peer[idx];
    __nv_bfloat16 *l = (__nv_bfloat16*)&v_local;
    __nv_bfloat16 *p = (__nv_bfloat16*)&v_peer;
    __nv_bfloat16 out[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = __hadd(l[i], p[i]);
    }
    *(uint4*)&local[idx] = *(uint4*)out;
}

int main() {
    int n; cudaGetDeviceCount(&n);
    if (n < 2) { printf("Need 2 GPUs\n"); return 1; }
    cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);

    size_t bytes = 1ull * 1024 * 1024 * 1024;  // 1 GB BF16 buffer
    int N = bytes / sizeof(__nv_bfloat16);
    int blocks = bytes / (256 * 16);  // 256 thr × 16 B per thread

    __nv_bfloat16 *d0, *d1;
    cudaSetDevice(0); cudaMalloc(&d0, bytes); cudaMemset(d0, 0x42, bytes);
    cudaSetDevice(1); cudaMalloc(&d1, bytes); cudaMemset(d1, 0x33, bytes);

    cudaSetDevice(0);
    cudaStream_t s0, s1;
    cudaStreamCreate(&s0);
    cudaSetDevice(1); cudaStreamCreate(&s1);

    cudaSetDevice(0);
    cudaEvent_t e0a, e0b; cudaEventCreate(&e0a); cudaEventCreate(&e0b);
    cudaSetDevice(1);
    cudaEvent_t e1a, e1b; cudaEventCreate(&e1a); cudaEventCreate(&e1b);

    auto run = [&]() {
        cudaSetDevice(0);
        allreduce_pair<<<blocks, 256, 0, s0>>>(d0, d1, N);
        cudaSetDevice(1);
        allreduce_pair<<<blocks, 256, 0, s1>>>(d1, d0, N);
    };

    // Warmup
    for (int i = 0; i < 5; i++) run();
    cudaSetDevice(0); cudaDeviceSynchronize();
    cudaSetDevice(1); cudaDeviceSynchronize();

    float best0 = 1e30f, best1 = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaSetDevice(0);
        cudaEventRecord(e0a, s0);
        cudaSetDevice(1);
        cudaEventRecord(e1a, s1);
        run();
        cudaSetDevice(0);
        cudaEventRecord(e0b, s0);
        cudaSetDevice(1);
        cudaEventRecord(e1b, s1);
        cudaEventSynchronize(e0b);
        cudaEventSynchronize(e1b);
        float ms0, ms1;
        cudaEventElapsedTime(&ms0, e0a, e0b);
        cudaEventElapsedTime(&ms1, e1a, e1b);
        if (ms0 < best0) best0 = ms0;
        if (ms1 < best1) best1 = ms1;
    }

    // Each GPU READ B from peer, WROTE B local
    // NVLink traffic = B per GPU = aggregate 2B over 2 GPUs (one direction)
    // OR each direction has B → bidir traffic 2B at 1500 GB/s peak = B/750
    double sol_us_unidir = (double)bytes / 783e9 * 1e6;
    double sol_us_bidir = (double)bytes / 750e9 * 1e6;
    double r0 = bytes / (best0/1000) / 1e9;
    double r1 = bytes / (best1/1000) / 1e9;

    printf("# 2-GPU all-reduce, %.0f MB BF16 buffer\n", bytes/1e6);
    printf("  GPU 0: %.4f ms = %.0f GB/s peer-read rate\n", best0, r0);
    printf("  GPU 1: %.4f ms = %.0f GB/s peer-read rate\n", best1, r1);
    printf("  Wall time (max): %.4f ms\n", best0 > best1 ? best0 : best1);
    printf("  SoL (unidir 783 GB/s): %.1f us; ratio %.2fx\n",
           sol_us_unidir, (best0 > best1 ? best0 : best1) * 1000 / sol_us_unidir);
    printf("  SoL (bidir 1500 GB/s/750 each-way): %.1f us; ratio %.2fx\n",
           sol_us_bidir, (best0 > best1 ? best0 : best1) * 1000 / sol_us_bidir);
    return 0;
}
