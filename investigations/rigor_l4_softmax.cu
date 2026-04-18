// L4 RIGOR: optimal BF16 row-softmax
// THEORETICAL: read N + write N bytes minimum (single-pass online softmax).
// Row of 4096 BF16 = 8 KB → 16 KB R+W. With 256K rows = 2 GB R + 2 GB W = 4 GB total.
// At HBM 7.30 TB/s peak: 548 us minimum.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Single-pass online softmax: each block does ONE row.
// Block size 256. Each thread reads ROW_LEN/256 elements.
// Compute running max + scaled sum, then second pass for normalize.
//
// Better: TWO-pass (split row across threads): read row to compute max,
// then read again to compute sum, then write normalized.
//
// For ROW_LEN=4096, 256 thr/block: each thread handles 16 elements.

extern "C" __launch_bounds__(256, 4) __global__ void softmax_2pass(
    const __nv_bfloat16 *x, __nv_bfloat16 *y, int row_len)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    const __nv_bfloat16 *row_x = x + row * row_len;
    __nv_bfloat16 *row_y = y + row * row_len;

    extern __shared__ float smem[];  // [0..255] for max reduce, [0..255] for sum

    // Pass 1: per-thread max
    int per_thread = row_len / 256;
    float m = -1e30f;
    for (int i = 0; i < per_thread; i++) {
        float v = __bfloat162float(row_x[i * 256 + t]);
        m = fmaxf(m, v);
    }
    // warp shfl reduce
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    // block reduce via shared
    int warp = t >> 5;
    int lane = t & 31;
    if (lane == 0) smem[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (lane < blockDim.x / 32) ? smem[lane] : -1e30f;
        for (int o = 4; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) smem[0] = m;
    }
    __syncthreads();
    float row_max = smem[0];

    // Pass 2: compute sum of exp(x - max)
    float s = 0.f;
    for (int i = 0; i < per_thread; i++) {
        s += __expf(__bfloat162float(row_x[i * 256 + t]) - row_max);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) smem[warp + 8] = s;
    __syncthreads();
    if (warp == 0) {
        s = (lane < blockDim.x / 32) ? smem[lane + 8] : 0.f;
        for (int o = 4; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) smem[1] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem[1];

    // Pass 3: write normalized
    for (int i = 0; i < per_thread; i++) {
        float v = __bfloat162float(row_x[i * 256 + t]);
        row_y[i * 256 + t] = __float2bfloat16(__expf(v - row_max) * inv_sum);
    }
}

int main() {
    cudaSetDevice(0);
    int row_len = 4096;
    int n_rows = 256 * 1024;  // 256K rows × 4096 BF16 = 2 GB
    size_t bytes_per_buf = (size_t)n_rows * row_len * sizeof(__nv_bfloat16);
    printf("# rows=%d, row_len=%d, total %.1f GB R + %.1f GB W (= %.1f us at 7300 GB/s peak)\n",
           n_rows, row_len, bytes_per_buf/1e9, bytes_per_buf/1e9,
           (double)bytes_per_buf*2 / 7300e9 * 1e6);

    __nv_bfloat16 *d_x; cudaMalloc(&d_x, bytes_per_buf);
    __nv_bfloat16 *d_y; cudaMalloc(&d_y, bytes_per_buf);

    // Init
    __nv_bfloat16 *h = (__nv_bfloat16*)malloc(bytes_per_buf);
    srand(42);
    for (size_t i = 0; i < (size_t)n_rows * row_len; i++)
        h[i] = __float2bfloat16(((float)rand()/RAND_MAX) * 4 - 2);
    cudaMemcpy(d_x, h, bytes_per_buf, cudaMemcpyHostToDevice);
    free(h);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int threads = 256;
    int smem_bytes = 16 * sizeof(float);

    auto bench = [&](const char* label) {
        for (int i = 0; i < 3; i++)
            softmax_2pass<<<n_rows, threads, smem_bytes>>>(d_x, d_y, row_len);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            softmax_2pass<<<n_rows, threads, smem_bytes>>>(d_x, d_y, row_len);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        size_t total_bytes = bytes_per_buf * 2;  // 1× R + 1× W (2-pass = 2R + 1W actually = 3*N)
        size_t actual_bytes = bytes_per_buf * 3;  // pass1 read + pass2 read + pass3 write
        double effective_gbs = total_bytes / (best/1000) / 1e9;
        double actual_gbs = actual_bytes / (best/1000) / 1e9;
        double sol_us = (double)bytes_per_buf*2 / 7300e9 * 1e6;
        printf("  %s: %.4f ms\n", label, best);
        printf("    Effective (R+W counted once): %.1f GB/s = %.1f%% of HBM peak\n",
               effective_gbs, effective_gbs/7300*100);
        printf("    Actual traffic (3× pass): %.1f GB/s = %.1f%% of HBM peak\n",
               actual_gbs, actual_gbs/7300*100);
        printf("    SoL bound (single pass R+W): %.1f us; ratio %.2fx slower\n",
               sol_us, best * 1000 / sol_us);
    };

    bench("softmax_2pass (full row in block)");

    return 0;
}
