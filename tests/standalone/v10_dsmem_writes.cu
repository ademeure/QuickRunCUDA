// V10: DSMEM write BW — st.shared::cluster.u32
#include <cuda_runtime.h>
#include <cstdio>

#define SMEM_W 2048

template<int CX, int NL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(128, 1)
void write_k(unsigned int* out, int iters, int seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    // Init
    for (int i = tid; i < SMEM_W; i += blockDim.x) smem[i] = (unsigned)i;
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

    unsigned base = peer_base + ((tid * 4) & (SMEM_W * 4 - 256));
    unsigned val = (unsigned)(tid + seed);

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        val++;  // Vary to prevent compiler from collapsing writes
        #pragma unroll
        for (int k = 0; k < NL; k++) {
            switch (k % 8) {
                case 0: asm volatile("st.shared::cluster.u32 [%0+0],   %1;" :: "r"(base), "r"(val)); break;
                case 1: asm volatile("st.shared::cluster.u32 [%0+32],  %1;" :: "r"(base), "r"(val)); break;
                case 2: asm volatile("st.shared::cluster.u32 [%0+64],  %1;" :: "r"(base), "r"(val)); break;
                case 3: asm volatile("st.shared::cluster.u32 [%0+96],  %1;" :: "r"(base), "r"(val)); break;
                case 4: asm volatile("st.shared::cluster.u32 [%0+128], %1;" :: "r"(base), "r"(val)); break;
                case 5: asm volatile("st.shared::cluster.u32 [%0+160], %1;" :: "r"(base), "r"(val)); break;
                case 6: asm volatile("st.shared::cluster.u32 [%0+192], %1;" :: "r"(base), "r"(val)); break;
                case 7: asm volatile("st.shared::cluster.u32 [%0+224], %1;" :: "r"(base), "r"(val)); break;
            }
        }
    }
    // Anti-DCE
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");
    out[gtid] = smem[tid & (SMEM_W - 1)];
}

template<int CX, int NL>
void run(int blocks, int iters, const char* label) {
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

    for (int i = 0; i < 3; i++) cudaLaunchKernelEx(&cfg, write_k<CX, NL>, d_out, iters, 42);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 5; i++) cudaLaunchKernelEx(&cfg, write_k<CX, NL>, d_out, iters, 42);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= 5;

    double bytes = (double)blocks * 128 * iters * NL * 4;
    double tbs = bytes / 1e12 / (ms / 1000.0);
    printf("  %s: %.2f ms → %.2f TB/s\n", label, ms, tbs);
    cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

int main() {
    cudaSetDevice(0);
    printf("=== DSMEM write BW (st.shared::cluster) ===\n");
    int blocks = 144;
    int iters = 10000;

    printf("\n--- Cluster size sweep (NL=8) ---\n");
    run<2, 8>(blocks, iters, "Cluster=2 NL=8");
    run<4, 8>(blocks, iters, "Cluster=4 NL=8");
    run<8, 8>(blocks, iters, "Cluster=8 NL=8");

    printf("\n--- Cluster=2 ILP sweep ---\n");
    run<2, 1>(blocks, iters, "NL=1 ");
    run<2, 4>(blocks, iters, "NL=4 ");
    run<2, 8>(blocks, iters, "NL=8 ");
    run<2, 16>(blocks, iters, "NL=16");

    return 0;
}
