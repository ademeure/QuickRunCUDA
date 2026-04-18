// NINJA softmax v3: split row across cluster of 2 blocks
// Each block reads HALF the row, exchanges max+sum via cluster, writes its half
// Total HBM: 1× R + 1× W (= SoL bound for 2-pass softmax)

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace cg = cooperative_groups;

// 2 blocks per row, each handles half. Cluster-shared comm.
extern "C" __launch_bounds__(256, 4) __global__ void
__cluster_dims__(2, 1, 1)
softmax_cluster(const __nv_bfloat16 *x, __nv_bfloat16 *y, int row_len)
{
    int row = blockIdx.x / 2;
    int blk_in_row = blockIdx.x & 1;
    int half = row_len / 2;
    int t = threadIdx.x;
    int lane = t & 31;

    const __nv_bfloat16 *my_x = x + row * row_len + blk_in_row * half;
    __nv_bfloat16 *my_y = y + row * row_len + blk_in_row * half;

    __shared__ float smem[16];  // 8 warp results (max), 8 (sum)
    __shared__ float row_max, row_sum;

    auto cluster = cg::this_cluster();

    // Pass 1: per-thread max (uint4 vec)
    int per_thread = half / 256;  // 2048/256 = 8 elements per thread
    int my_off = t * per_thread;
    float m = -1e30f;
    uint4 v = *(uint4*)&my_x[my_off];
    const __nv_bfloat16 *p = (const __nv_bfloat16*)&v;
    #pragma unroll
    for (int i = 0; i < 8; i++) m = fmaxf(m, __bfloat162float(p[i]));

    // Warp reduce
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    int warp = t >> 5;
    if (lane == 0) smem[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (lane < 8) ? smem[lane] : -1e30f;
        for (int o = 4; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) smem[0] = m;  // local block max
    }
    __syncthreads();

    // Now exchange via cluster: get peer's max via DSMEM
    cluster.sync();  // all threads must participate
    if (t == 0) {
        float *peer_smem = (float*)cluster.map_shared_rank(smem, 1 - blk_in_row);
        float my_max = smem[0];
        float peer_max = peer_smem[0];
        row_max = fmaxf(my_max, peer_max);
    }
    __syncthreads();
    float rm = row_max;

    // Pass 2: sum of exp using row_max
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; i++) s += __expf(__bfloat162float(p[i]) - rm);
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) smem[warp + 8] = s;
    __syncthreads();
    if (warp == 0) {
        s = (lane < 8) ? smem[lane + 8] : 0.f;
        for (int o = 4; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) smem[1] = s;
    }
    __syncthreads();

    cluster.sync();
    if (t == 0) {
        float *peer_smem = (float*)cluster.map_shared_rank(smem, 1 - blk_in_row);
        float my_sum = smem[1];
        float peer_sum = peer_smem[1];
        row_sum = my_sum + peer_sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / row_sum;

    // Pass 3: write normalized (only need to read x once more or use cached `p`)
    __nv_bfloat16 outbuf[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        outbuf[i] = __float2bfloat16(__expf(__bfloat162float(p[i]) - rm) * inv_sum);
    }
    *(uint4*)&my_y[my_off] = *(uint4*)outbuf;
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

    int blocks = n_rows * 2;  // 2 blocks per row
    int threads = 256;

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cudaLaunchAttribute attr = {};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;

    auto launch = [&]() {
        cudaLaunchKernelEx(&cfg, softmax_cluster, d_x, d_y, row_len);
    };

    for (int i = 0; i < 3; i++) launch();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ERR: %s\n", cudaGetErrorString(err)); return 1; }

    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        launch();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    size_t actual = bytes * 2;  // 1R + 1W (single-pass via cached register vector!)
    double a_gbs = actual / (best/1000) / 1e9;
    double sol = (double)bytes*2/7300e9*1e6;
    printf("# softmax_cluster (2 blocks per row): %.4f ms\n", best);
    printf("  Actual traffic (1R+1W): %.0f GB/s = %.1f%% of HBM peak\n",
           a_gbs, a_gbs/7300*100);
    printf("  SoL bound: %.1f us; ratio %.2fx slower\n", sol, best*1000/sol);

    return 0;
}
