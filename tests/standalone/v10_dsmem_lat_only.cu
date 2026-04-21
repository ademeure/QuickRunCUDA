// V10: minimal cluster=8 DSMEM latency test (chained ld.shared::cluster)
#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define SMEM_W 256
#define CHAIN_LEN 1024

__global__ __cluster_dims__(8, 1, 1) __launch_bounds__(32, 1)
void dsmem_lat_chain(unsigned long long* out) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;

    // Init: smem[i] = byte offset of next (matches bench_dsmem_definitive)
    for (int i = tid; i < SMEM_W; i += 32) {
        smem[i] = (unsigned)(((i + 32) & (SMEM_W - 1)) * 4);  // byte offset
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (my_cta + 1u) & 7;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    // cur = byte offset, starts at tid*4 (within smem)
    unsigned cur = (unsigned)(tid * 4) & ((SMEM_W - 1) * 4);
    unsigned long long t0, t1;
    if (tid == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Chain: cur = peer_smem[cur/4], which returns BYTE offset
    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur) : "r"(peer_base + cur) : "memory");
    }

    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        if (my_cta == 0 && blockIdx.x == 0) {
            out[0] = t1 - t0;
            ((unsigned*)out)[2] = cur;
        }
    }
}

__global__ __launch_bounds__(32, 1)
void smem_lat_chain(unsigned long long* out) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32)
        smem[i] = (unsigned)((i + 32) & (SMEM_W - 1));
    __syncthreads();
    if (tid != 0) return;
    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned idx = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(idx) : "r"(local_base + idx * 4));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (blockIdx.x == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = idx;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 16));

    // Local baseline
    smem_lat_chain<<<1, 32>>>(d_out);
    CK(cudaDeviceSynchronize());
    smem_lat_chain<<<1, 32>>>(d_out);
    CK(cudaDeviceSynchronize());
    unsigned long long c_local;
    cudaMemcpy(&c_local, d_out, 8, cudaMemcpyDeviceToHost);
    printf("Local SMEM chain: %.2f cy/load\n", (double)c_local / CHAIN_LEN);

    // Cluster=8 DSMEM
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(8, 1, 1);
    cfg.blockDim = dim3(32, 1, 1);
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 8;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    CK(cudaLaunchKernelEx(&cfg, dsmem_lat_chain, d_out));
    CK(cudaDeviceSynchronize());
    CK(cudaLaunchKernelEx(&cfg, dsmem_lat_chain, d_out));
    CK(cudaDeviceSynchronize());
    unsigned long long c_dsmem;
    cudaMemcpy(&c_dsmem, d_out, 8, cudaMemcpyDeviceToHost);
    printf("Cluster=8 DSMEM chain: %.2f cy/load\n", (double)c_dsmem / CHAIN_LEN);
    printf("DSMEM/Local ratio: %.2fx\n", (double)c_dsmem / c_local);

    cudaFree(d_out);
    return 0;
}
