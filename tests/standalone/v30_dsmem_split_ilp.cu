// V30: Per-CTA BW — is it limited by reader issue rate or peer serving rate?
// Test: CTA 0 splits ILP chains across N peers.
// If reader-issue-limited → total BW same as single-peer
// If peer-serving-limited → total BW scales with peers

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 8
#define CL 5

// CTA 0 reads ILP_TOTAL chains split across N_PEERS peers
template<int N_PEERS, int ILP_PER_PEER>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void split_peer_reads(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);

    // Compute N_PEERS peer bases
    unsigned peer_bases[N_PEERS];
    #pragma unroll
    for (int p = 0; p < N_PEERS; p++) {
        unsigned pb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(pb) : "r"(local_base), "r"(p + 1));
        peer_bases[p] = pb;
    }

    // ILP chains: N_PEERS × ILP_PER_PEER total
    const int TOTAL = N_PEERS * ILP_PER_PEER;
    unsigned c[32];
    #pragma unroll
    for (int k = 0; k < TOTAL; k++) c[k] = (tid * 4u + k * 32u) & (SMEM_W*4 - 1);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (my_cta == 0) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            // Interleave: for each chain, use its assigned peer
            #pragma unroll
            for (int k = 0; k < TOTAL; k++) {
                int p = k % N_PEERS;  // which peer
                asm volatile("ld.shared::cluster.u32 %0, [%1];"
                             : "=r"(c[k]) : "r"(peer_bases[p] + c[k]) : "memory");
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        unsigned acc = 0;
        for (int k = 0; k < TOTAL; k++) acc ^= c[k];
        ((unsigned*)out)[2] = acc;
    }
}

template<typename K, typename... Args>
static int avg_cy_t(K kernel, int N, double* cy_out, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CX, 1, 1);
    cfg.blockDim = dim3(32, 1, 1);
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CX;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    unsigned long long* d_out;
    cudaMalloc(&d_out, 16);
    double sum = 0; int got = 0;
    for (int r = 0; r < N * 2 && got < N; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, d_out, args...);
        if (e) { cudaGetLastError(); continue; }
        e = cudaDeviceSynchronize();
        if (e) { cudaGetLastError(); continue; }
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        sum += (double)cy;
        got++;
    }
    cudaFree(d_out);
    if (got > 0) { *cy_out = sum / got; return got; }
    return 0;
}

int main() {
    CK(cudaSetDevice(0));
    const double CLOCK_GHZ = 1.920;
    double cy;

    printf("=== V30 DSMEM split-ILP — is BW reader-limited or peer-limited? ===\n\n");
    printf("CTA 0 reads from N_PEERS peers with ILP_PER_PEER chains each.\n");
    printf("Total chains = N_PEERS × ILP_PER_PEER. Only CTA 0 active.\n\n");
    printf("  N_peers × ILP/peer  Total_ILP  cy_per_CTA  cy/load  per-CTA BW\n");

    #define S(N, I) { \
        int got = avg_cy_t(split_peer_reads<N, I>, 15, &cy, 42u); \
        if (got > 0) { \
            double loads = 32.0 * N * I * CL; \
            double t = cy / CLOCK_GHZ / 1e9; \
            printf("  %d      × %-2d         %3d       %7.0f   %5.2f    %.2f GB/s\n", \
                   N, I, N*I, cy, cy/loads, loads*4/t/1e9); \
        } }
    S(1, 1); S(1, 4); S(1, 8); S(1, 16);
    printf("  --- splitting 8 total ILP chains across peers ---\n");
    S(2, 4); S(4, 2); S(8, 1);
    printf("  --- splitting 16 total ILP chains across peers ---\n");
    S(2, 8); S(4, 4); S(7, 2);
    #undef S

    return 0;
}
