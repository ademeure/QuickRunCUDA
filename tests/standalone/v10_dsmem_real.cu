// V10 REAL: DSMEM BW with verified dependency chain (no DCE)
// Each load's result is mask-folded into the NEXT load's byte offset.
#include <cuda_runtime.h>
#include <cstdio>

#define SMEM_W 2048
// SMEM_W * 4 = 8192 bytes. Mask to stay in range with offset.
#define BOUND_MASK (8192 - 256)

template<int CX>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(128, 1)
void k(unsigned int* out, int iters, int seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    // Init smem so chain values stay bounded
    for (int i = tid; i < SMEM_W; i += blockDim.x) {
        smem[i] = (unsigned)(((i * 37u + seed) & BOUND_MASK) & ~3u);
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (my_cta + 1u) % CX;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    // Each lane has a chain. Start offset is tid*4 (wrapped).
    unsigned cur = ((unsigned)tid * 4) & BOUND_MASK;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        unsigned r;
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(r) : "r"(peer_base + cur) : "memory");
        cur = r;  // TRUE dependency: next load's addr depends on this value
    }

    out[gtid] = cur;  // Anti-DCE via real write
}

template<int CX>
void run(int blocks, int iters) {
    unsigned int* d_out;
    cudaMalloc(&d_out, 16 * 1024 * 1024);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks, 1, 1);
    cfg.blockDim = dim3(128, 1, 1);
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CX;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    // Warmup
    for (int i = 0; i < 3; i++) cudaLaunchKernelEx(&cfg, k<CX>, d_out, iters, 42);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    cudaLaunchKernelEx(&cfg, k<CX>, d_out, iters, 42);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);

    // Bytes: blocks × threads × iters × 4 (each load is 4 B)
    double bytes = (double)blocks * 128 * iters * 4;
    double tbs = bytes / 1e12 / (ms / 1000.0);
    printf("  Cluster=%d chained DSMEM: %.3f ms → %.2f TB/s (1 load/iter latency-bound)\n", CX, ms, tbs);
    cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

int main() {
    cudaSetDevice(0);
    printf("=== DSMEM REAL BW test (dep-chained, ncu-verify) ===\n");
    int blocks = 144;
    int iters = 100000;
    run<2>(blocks, iters);
    run<4>(blocks, iters);
    run<8>(blocks, iters);
    return 0;
}
