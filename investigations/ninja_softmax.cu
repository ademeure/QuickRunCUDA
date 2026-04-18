// NINJA softmax: try fewer rows but more parallelism per row, larger blocks
// Original L4: 1 block per row, 256 thr/block, 16 elem/thread
// Try: 1024 thr/block (4 warps), 4 elem/thread = same row size but more warps in flight

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" __launch_bounds__(1024, 2) __global__ void softmax_v1024(
    const __nv_bfloat16 *x, __nv_bfloat16 *y, int row_len)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    const __nv_bfloat16 *row_x = x + row * row_len;
    __nv_bfloat16 *row_y = y + row * row_len;

    extern __shared__ float smem[];
    int per_thread = row_len / 1024;  // 4096/1024 = 4

    // Pass 1: max
    float m = -1e30f;
    for (int i = 0; i < per_thread; i++) {
        m = fmaxf(m, __bfloat162float(row_x[i * 1024 + t]));
    }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    int warp = t >> 5, lane = t & 31;
    if (lane == 0) smem[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (lane < 32) ? smem[lane] : -1e30f;
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) smem[0] = m;
    }
    __syncthreads();
    float row_max = smem[0];

    // Pass 2: sum of exp
    float s = 0.f;
    for (int i = 0; i < per_thread; i++) {
        s += __expf(__bfloat162float(row_x[i * 1024 + t]) - row_max);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) smem[warp + 32] = s;
    __syncthreads();
    if (warp == 0) {
        s = (lane < 32) ? smem[lane + 32] : 0.f;
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) smem[1] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem[1];

    // Pass 3: write
    for (int i = 0; i < per_thread; i++) {
        float v = __bfloat162float(row_x[i * 1024 + t]);
        row_y[i * 1024 + t] = __float2bfloat16(__expf(v - row_max) * inv_sum);
    }
}

// Try smaller row in larger block — better warp parallelism
extern "C" __launch_bounds__(512, 4) __global__ void softmax_v512(
    const __nv_bfloat16 *x, __nv_bfloat16 *y, int row_len)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    const __nv_bfloat16 *row_x = x + row * row_len;
    __nv_bfloat16 *row_y = y + row * row_len;

    extern __shared__ float smem[];
    int per_thread = row_len / 512;  // 4096/512 = 8

    float m = -1e30f;
    // Use uint4 vectorized read of 8 BF16 = 1 thread reads its 8 elems in 1 load
    int my_off = t * per_thread;
    if (per_thread == 8) {
        uint4 v = *(uint4*)&row_x[my_off];
        const __nv_bfloat16 *p = (const __nv_bfloat16*)&v;
        #pragma unroll
        for (int i = 0; i < 8; i++) m = fmaxf(m, __bfloat162float(p[i]));
    } else {
        for (int i = 0; i < per_thread; i++)
            m = fmaxf(m, __bfloat162float(row_x[my_off + i]));
    }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    int warp = t >> 5, lane = t & 31;
    if (lane == 0) smem[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (lane < 16) ? smem[lane] : -1e30f;
        for (int o = 8; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) smem[0] = m;
    }
    __syncthreads();
    float row_max = smem[0];

    float s = 0.f;
    if (per_thread == 8) {
        uint4 v = *(uint4*)&row_x[my_off];
        const __nv_bfloat16 *p = (const __nv_bfloat16*)&v;
        #pragma unroll
        for (int i = 0; i < 8; i++) s += __expf(__bfloat162float(p[i]) - row_max);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) smem[warp + 16] = s;
    __syncthreads();
    if (warp == 0) {
        s = (lane < 16) ? smem[lane + 16] : 0.f;
        for (int o = 8; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) smem[1] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem[1];

    if (per_thread == 8) {
        uint4 v = *(uint4*)&row_x[my_off];
        const __nv_bfloat16 *p = (const __nv_bfloat16*)&v;
        __nv_bfloat16 outbuf[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            outbuf[i] = __float2bfloat16(__expf(__bfloat162float(p[i]) - row_max) * inv_sum);
        }
        *(uint4*)&row_y[my_off] = *(uint4*)outbuf;
    }
}

int main() {
    cudaSetDevice(0);
    int row_len = 4096;
    int n_rows = 256 * 1024;
    size_t bytes = (size_t)n_rows * row_len * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_x, *d_y;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    __nv_bfloat16 *h = (__nv_bfloat16*)malloc(bytes);
    srand(42);
    for (size_t i = 0; i < (size_t)n_rows*row_len; i++)
        h[i] = __float2bfloat16(((float)rand()/RAND_MAX)*4-2);
    cudaMemcpy(d_x, h, bytes, cudaMemcpyHostToDevice);
    free(h);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](auto launch, const char* label) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 10; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        size_t actual = bytes * 3;  // 2R + 1W (assuming compiler caches reads)
        size_t effective = bytes * 2;  // SoL bound
        double a_gbs = actual / (best/1000) / 1e9;
        double e_gbs = effective / (best/1000) / 1e9;
        double sol = (double)bytes*2/7300e9*1e6;
        printf("  %s: %.4f ms = actual %.0f GB/s = %.1f%% HBM (SoL %.1f us, %.2fx)\n",
               label, best, a_gbs, a_gbs/7300*100, sol, best*1000/sol);
    };

    bench([&]{ softmax_v1024<<<n_rows, 1024, 64*sizeof(float)>>>(d_x, d_y, row_len); }, "v1024 (1024 thr)");
    bench([&]{ softmax_v512<<<n_rows, 512, 32*sizeof(float)>>>(d_x, d_y, row_len); }, "v512 (uint4 vec) ");

    return 0;
}
