// V10: Cluster=2 DSMEM — push ILP higher
#include <cuda_runtime.h>
#include <cstdio>

#define SMEM_W 2048

template<int NL>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1)
void k(unsigned int* out, int iters, int seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < SMEM_W; i += blockDim.x) smem[i] = (unsigned)i ^ (unsigned)seed;
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (my_cta + 1u) & 1;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    unsigned base = peer_base + ((tid * 4) & (SMEM_W * 4 - 256));
    unsigned accs[32] = {0};

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int k = 0; k < NL; k++) {
            unsigned r;
            // Compile-time modulo 8 for the offset, spread over 0..224 B
            switch (k % 8) {
                case 0: asm volatile("ld.shared::cluster.u32 %0, [%1+0];"   : "=r"(r) : "r"(base)); break;
                case 1: asm volatile("ld.shared::cluster.u32 %0, [%1+32];"  : "=r"(r) : "r"(base)); break;
                case 2: asm volatile("ld.shared::cluster.u32 %0, [%1+64];"  : "=r"(r) : "r"(base)); break;
                case 3: asm volatile("ld.shared::cluster.u32 %0, [%1+96];"  : "=r"(r) : "r"(base)); break;
                case 4: asm volatile("ld.shared::cluster.u32 %0, [%1+128];" : "=r"(r) : "r"(base)); break;
                case 5: asm volatile("ld.shared::cluster.u32 %0, [%1+160];" : "=r"(r) : "r"(base)); break;
                case 6: asm volatile("ld.shared::cluster.u32 %0, [%1+192];" : "=r"(r) : "r"(base)); break;
                case 7: asm volatile("ld.shared::cluster.u32 %0, [%1+224];" : "=r"(r) : "r"(base)); break;
            }
            accs[k] += r;
        }
    }
    unsigned sum = 0;
    for (int k = 0; k < NL; k++) sum ^= accs[k];
    out[gtid] = sum;
}

template<int NL>
void run(int blocks, int iters) {
    unsigned int* d_out;
    cudaMalloc(&d_out, 16 * 1024 * 1024);
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

    for (int i = 0; i < 3; i++) cudaLaunchKernelEx(&cfg, k<NL>, d_out, iters, 42);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 5; i++) cudaLaunchKernelEx(&cfg, k<NL>, d_out, iters, 42);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= 5;

    double bytes = (double)blocks * 128 * iters * NL * 4;
    double tbs = bytes / 1e12 / (ms / 1000.0);
    printf("  NL=%2d: %.2f ms → %.2f TB/s\n", NL, ms, tbs);
    cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

int main() {
    cudaSetDevice(0);
    printf("=== Cluster=2 DSMEM ILP scan ===\n");
    int blocks = 144;
    int iters = 10000;
    run<1>(blocks, iters);
    run<2>(blocks, iters);
    run<4>(blocks, iters);
    run<8>(blocks, iters);
    run<16>(blocks, iters);
    run<32>(blocks, iters);
    return 0;
}
