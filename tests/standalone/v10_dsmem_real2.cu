// V10 REAL v2: DSMEM BW — base varies each iter, results accumulate, NO dep chain
#include <cuda_runtime.h>
#include <cstdio>

#define SMEM_W 2048

template<int CX>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(128, 1)
void k(unsigned int* out, int iters, int seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < SMEM_W; i += blockDim.x)
        smem[i] = (unsigned)i ^ (unsigned)seed;
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

    unsigned acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    unsigned acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // base VARIES with i — forces distinct load each iter
        unsigned base = peer_base + (((tid * 4) + (i * 4)) & 0x1FE0);  // step 4 B, mask 8 KB-32

        unsigned r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("ld.shared::cluster.u32 %0, [%1+0];"   : "=r"(r0) : "r"(base));
        asm volatile("ld.shared::cluster.u32 %0, [%1+4];"   : "=r"(r1) : "r"(base));
        asm volatile("ld.shared::cluster.u32 %0, [%1+8];"   : "=r"(r2) : "r"(base));
        asm volatile("ld.shared::cluster.u32 %0, [%1+12];"  : "=r"(r3) : "r"(base));
        asm volatile("ld.shared::cluster.u32 %0, [%1+16];"  : "=r"(r4) : "r"(base));
        asm volatile("ld.shared::cluster.u32 %0, [%1+20];"  : "=r"(r5) : "r"(base));
        asm volatile("ld.shared::cluster.u32 %0, [%1+24];"  : "=r"(r6) : "r"(base));
        asm volatile("ld.shared::cluster.u32 %0, [%1+28];"  : "=r"(r7) : "r"(base));
        acc0 += r0; acc1 += r1; acc2 += r2; acc3 += r3;
        acc4 += r4; acc5 += r5; acc6 += r6; acc7 += r7;
    }
    out[gtid] = acc0 ^ acc1 ^ acc2 ^ acc3 ^ acc4 ^ acc5 ^ acc6 ^ acc7;
}

int main() {
    cudaSetDevice(0);
    unsigned int* d_out;
    cudaMalloc(&d_out, 16 * 1024 * 1024);

    int blocks = 144;
    int iters = 10000;

    // Cluster=2 test
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks, 1, 1);
    cfg.blockDim = dim3(128, 1, 1);
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    // Warmup
    for (int i = 0; i < 3; i++) cudaLaunchKernelEx(&cfg, k<2>, d_out, iters, 42);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    cudaLaunchKernelEx(&cfg, k<2>, d_out, iters, 42);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);

    // 8 loads per iter × iters × threads × 4 B
    double bytes = (double)blocks * 128 * iters * 8 * 4;
    double tbs = bytes / 1e12 / (ms / 1000.0);
    printf("Cluster=2, 8-load/iter varying base, %d iters: %.3f ms → %.2f TB/s\n",
           iters, ms, tbs);

    cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
