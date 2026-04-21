// V10: reproduce V8 bench_dsmem_definitive kernel directly (to verify works standalone)
#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define CLUSTER_X 8
#define LAT_ITERS 1024
#define SMEM_WORDS 256

__global__ __cluster_dims__(CLUSTER_X, 1, 1) __launch_bounds__(128, 1)
void kernel_v8(unsigned long long* C, int seed) {
    __shared__ unsigned smem[SMEM_WORDS];
    for (int i = threadIdx.x; i < SMEM_WORDS; i += blockDim.x) {
        unsigned v = (unsigned)i * 2654435761u ^ (unsigned)seed;
        smem[i] = ((v >> 5) & (SMEM_WORDS - 1u)) * 4u;
    }
    __syncthreads();

    unsigned my_cta, cluster_size;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_size));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    int tid = threadIdx.x;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (my_cta + 1u) % cluster_size;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    if (tid < 32) {
        unsigned rem_cur = ((unsigned)(tid * 31 + seed + 1) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
        unsigned long long t2, t3;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t2) :: "memory");
        #pragma unroll 1
        for (int i = 0; i < LAT_ITERS; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(rem_cur) : "r"(peer_base + rem_cur) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t3) :: "memory");
        if (tid == 0 && my_cta == 0) {
            C[0] = t3 - t2;
            ((unsigned*)C)[2] = rem_cur;
        }
    }
}

int main() {
    cudaSetDevice(0);
    unsigned long long* d_C;
    CK(cudaMalloc(&d_C, 16));

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CLUSTER_X, 1, 1);
    cfg.blockDim = dim3(128, 1, 1);
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_X;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    CK(cudaLaunchKernelEx(&cfg, kernel_v8, d_C, 42));
    CK(cudaDeviceSynchronize());
    CK(cudaLaunchKernelEx(&cfg, kernel_v8, d_C, 42));
    CK(cudaDeviceSynchronize());

    unsigned long long cycles;
    cudaMemcpy(&cycles, d_C, 8, cudaMemcpyDeviceToHost);
    printf("V8-repro cluster=8 DSMEM chained LDS: %.2f cy/load\n", (double)cycles / LAT_ITERS);

    cudaFree(d_C);
    return 0;
}
