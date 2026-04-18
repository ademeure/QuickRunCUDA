// NINJA absmax: apply 1-store-per-warp principle (small per-warp work)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Original L2: 592 blocks × 256 thr × stride loop. Achieves 92.3%
// Ninja: many more blocks, less per-warp work
__launch_bounds__(256, 8) __global__ void absmax_ninja(
    const __nv_bfloat16 *x, float *out, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    // Each warp handles 1024 B = 512 BF16 (matches HBM ninja burst)
    int base = warp_id * 512 + lane * 16;  // 32 lanes × 16 BF16/lane = 512 BF16

    if (base >= N) return;
    float local = 0.f;

    // 16 BF16 = 2 uint4 = 32 B per lane
    uint4 v0 = *(uint4*)&x[base];
    uint4 v1 = *(uint4*)&x[base + 8];
    const __nv_bfloat16 *p0 = (const __nv_bfloat16*)&v0;
    const __nv_bfloat16 *p1 = (const __nv_bfloat16*)&v1;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        local = fmaxf(local, fabsf(__bfloat162float(p0[j])));
        local = fmaxf(local, fabsf(__bfloat162float(p1[j])));
    }
    // warp shfl reduce
    for (int o = 16; o; o >>= 1)
        local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, o));
    if (lane == 0) atomicMax((int*)out, __float_as_int(local));
}

// Even more aggressive: each warp does just 1 uint4 load = 256 B
__launch_bounds__(256, 8) __global__ void absmax_xtreme(
    const __nv_bfloat16 *x, float *out, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int base = warp_id * 256 + lane * 8;
    if (base >= N) return;

    float local = 0.f;
    uint4 v = *(uint4*)&x[base];
    const __nv_bfloat16 *p = (const __nv_bfloat16*)&v;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        local = fmaxf(local, fabsf(__bfloat162float(p[j])));
    }
    for (int o = 16; o; o >>= 1)
        local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, o));
    if (lane == 0) atomicMax((int*)out, __float_as_int(local));
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 1ull * 1024 * 1024 * 1024;
    int N = bytes / sizeof(__nv_bfloat16);

    __nv_bfloat16 *h = (__nv_bfloat16*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) h[i] = __float2bfloat16(((float)rand()/RAND_MAX) * 2 - 1);
    h[N/2] = __float2bfloat16(7.5f);

    __nv_bfloat16 *d_x; cudaMalloc(&d_x, bytes);
    cudaMemcpy(d_x, h, bytes, cudaMemcpyHostToDevice);
    free(h);

    float *d_out; cudaMalloc(&d_out, sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch, const char* label) {
        cudaMemset(d_out, 0, sizeof(float));
        for (int i = 0; i < 5; i++) { cudaMemset(d_out, 0, sizeof(float)); launch(); }
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaMemset(d_out, 0, sizeof(float));
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        float r; cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost);
        double gbs = bytes / (best/1000) / 1e9;
        printf("  %s: %.4f ms = %.1f GB/s = %.2f%% spec  (result=%.4f)\n",
               label, best, gbs, gbs/7672*100, r);
    };

    // ninja: 1024 B per warp = 32 BF16 per thread × 16 (lanes×16BF16=512); blocks=N/512/8=N/4096 wait
    int warps_per_block = 8;
    int n_warps_ninja = (N + 511) / 512;
    int blocks_ninja = (n_warps_ninja + warps_per_block - 1) / warps_per_block;
    int n_warps_xtreme = (N + 255) / 256;
    int blocks_xtreme = (n_warps_xtreme + warps_per_block - 1) / warps_per_block;
    printf("# blocks ninja=%d xtreme=%d\n", blocks_ninja, blocks_xtreme);

    bench([&]{ absmax_ninja<<<blocks_ninja, 256>>>(d_x, d_out, N); }, "ninja  (512 BF16/warp)");
    bench([&]{ absmax_xtreme<<<blocks_xtreme, 256>>>(d_x, d_out, N); }, "xtreme (256 BF16/warp)");

    return 0;
}
