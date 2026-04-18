// NINJA absmax v3: try smaller per-warp work + tuned block count
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Each warp processes EXACTLY 1 KB (matches HBM ninja). 32 lanes × 16 BF16/lane = 512 BF16.
__launch_bounds__(256, 4) __global__ void absmax_1kb_per_warp(
    const __nv_bfloat16 *x, float *out, int N)
{
    __shared__ float smem[8];
    int t = threadIdx.x;
    int warp = t >> 5;
    int lane = t & 31;
    int warp_in_grid = blockIdx.x * 8 + warp;
    int total_warps = gridDim.x * 8;

    float local = 0.f;
    // Each warp covers chunks of 512 BF16 = 1 KB at stride
    for (int wbase = warp_in_grid * 512; wbase < N; wbase += total_warps * 512) {
        int my_off = wbase + lane * 16;
        if (my_off + 15 >= N) break;
        // 16 BF16 = 2 uint4 = 32 B per lane
        uint4 v0 = *(uint4*)&x[my_off];
        uint4 v1 = *(uint4*)&x[my_off + 8];
        const __nv_bfloat16 *p0 = (const __nv_bfloat16*)&v0;
        const __nv_bfloat16 *p1 = (const __nv_bfloat16*)&v1;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            local = fmaxf(local, fabsf(__bfloat162float(p0[j])));
            local = fmaxf(local, fabsf(__bfloat162float(p1[j])));
        }
    }
    for (int o = 16; o; o >>= 1)
        local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, o));
    if (lane == 0) smem[warp] = local;
    __syncthreads();
    if (warp == 0) {
        float bv = (lane < 8) ? smem[lane] : 0.f;
        for (int o = 4; o; o >>= 1)
            bv = fmaxf(bv, __shfl_xor_sync(0xffffffff, bv, o));
        if (lane == 0) atomicMax((int*)out, __float_as_int(bv));
    }
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

    auto bench = [&](int blocks) {
        cudaMemset(d_out, 0, 4);
        for (int i = 0; i < 5; i++) {
            cudaMemset(d_out, 0, 4);
            absmax_1kb_per_warp<<<blocks, 256>>>(d_x, d_out, N);
        }
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaMemset(d_out, 0, 4);
            cudaEventRecord(e0);
            absmax_1kb_per_warp<<<blocks, 256>>>(d_x, d_out, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        float r; cudaMemcpy(&r, d_out, 4, cudaMemcpyDeviceToHost);
        double gbs = bytes / (best/1000) / 1e9;
        printf("  blocks=%5d: %.4f ms = %.0f GB/s = %.2f%% spec (result=%.4f)\n",
               blocks, best, gbs, gbs/7672*100, r);
    };

    for (int b : {2368, 4736, 9472, 18944, 37888, 65536, 131072}) bench(b);
    return 0;
}
