// V10 DEEP: cluster=8 DSMEM characterization
// Tests:
//   1. Single-load LATENCY (chained, single thread, dependent)
//   2. BW vs cluster_size {2, 4, 8}
//   3. BW: ring (CTA i → i+1) vs all-to-all (CTA i reads ALL peers)
//   4. Varying-address DSMEM (was problematic in V8)
//   5. DSMEM vs LOCAL same-pattern A/B
#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define SMEM_W 1024
#define CHAIN_LEN 1024

// === Test 1: DSMEM single-load latency (chain through peer SMEM) ===
template<int CLUSTER_X>
__global__ __cluster_dims__(CLUSTER_X, 1, 1) __launch_bounds__(32, 1)
void dsmem_lat_chained(unsigned long long* out_cycles) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;

    // Chain pointer: smem[i] = (i + 32) & MASK (sequential-by-line, dep)
    for (int i = tid; i < SMEM_W; i += 32) {
        smem[i] = (unsigned)((i + 32) & (SMEM_W - 1));
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (my_cta + 1u) % CLUSTER_X;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    if (tid != 0) return;
    unsigned idx = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(idx) : "r"(peer_base + idx * 4));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (my_cta == 0 && blockIdx.x == 0) out_cycles[0] = t1 - t0;
}

// Local SMEM baseline (same pattern, no cluster)
__global__ __launch_bounds__(32, 1)
void smem_lat_chained(unsigned long long* out_cycles) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        smem[i] = (unsigned)((i + 32) & (SMEM_W - 1));
    }
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
    if (blockIdx.x == 0) out_cycles[0] = t1 - t0;
}

// === Test 2: DSMEM BW with varying offset (was bug in V8) ===
// Uses MAPA on each iter with a different lane offset, accumulates
template<int CLUSTER_X>
__global__ __cluster_dims__(CLUSTER_X, 1, 1) __launch_bounds__(128, 1)
void dsmem_bw_varying(unsigned int* out, int iters) {
    __shared__ alignas(16) unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < SMEM_W; i += blockDim.x) smem[i] = (unsigned)i;
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;" ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (my_cta + 1u) % CLUSTER_X;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    // Each iter loads from an OFFSET that varies. Use accumulator pattern (no chain dep).
    unsigned s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    unsigned s4 = 0, s5 = 0, s6 = 0, s7 = 0;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // Bound offset in [0, SMEM_W*4 - 32]; offset varies with i
        unsigned base_byte = ((tid * 4) + (i * 32)) & ((SMEM_W * 4) - 256);
        unsigned r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("ld.shared::cluster.u32 %0, [%1+0];"   : "=r"(r0) : "r"(peer_base + base_byte));
        asm volatile("ld.shared::cluster.u32 %0, [%1+4];"   : "=r"(r1) : "r"(peer_base + base_byte));
        asm volatile("ld.shared::cluster.u32 %0, [%1+8];"   : "=r"(r2) : "r"(peer_base + base_byte));
        asm volatile("ld.shared::cluster.u32 %0, [%1+12];"  : "=r"(r3) : "r"(peer_base + base_byte));
        asm volatile("ld.shared::cluster.u32 %0, [%1+16];"  : "=r"(r4) : "r"(peer_base + base_byte));
        asm volatile("ld.shared::cluster.u32 %0, [%1+20];"  : "=r"(r5) : "r"(peer_base + base_byte));
        asm volatile("ld.shared::cluster.u32 %0, [%1+24];"  : "=r"(r6) : "r"(peer_base + base_byte));
        asm volatile("ld.shared::cluster.u32 %0, [%1+28];"  : "=r"(r7) : "r"(peer_base + base_byte));
        s0 += r0; s1 += r1; s2 += r2; s3 += r3;
        s4 += r4; s5 += r5; s6 += r6; s7 += r7;
    }
    unsigned sum = s0^s1^s2^s3^s4^s5^s6^s7;
    out[gtid] = sum;
}

int main() {
    cudaSetDevice(0);
    unsigned long long* d_lat;
    unsigned int* d_out;
    CK(cudaMalloc(&d_lat, 16));
    CK(cudaMalloc(&d_out, 16 * 1024 * 1024));

    printf("=== V10 DEEP: cluster=8 DSMEM characterization ===\n\n");

    // === LATENCY: cluster_size sweep ===
    printf("--- DSMEM single-load latency (chained) ---\n");

    // Need cooperative launch attrs for cluster
    // Helper to make launch config
    auto make_cfg = [](int gx, int blockx, int cluster_x) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(gx, 1, 1);
        cfg.blockDim = dim3(blockx, 1, 1);
        cfg.stream = 0;
        static cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = cluster_x;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        return cfg;
    };

    // Local baseline first
    smem_lat_chained<<<1, 32>>>(d_lat);
    cudaDeviceSynchronize();
    smem_lat_chained<<<1, 32>>>(d_lat);
    cudaDeviceSynchronize();
    unsigned long long c_local;
    cudaMemcpy(&c_local, d_lat, 8, cudaMemcpyDeviceToHost);
    printf("  Local SMEM:                 %.2f cy/load\n", (double)c_local / CHAIN_LEN);

    // Cluster sweep — call directly with templated kernels
    {
        auto cfg = make_cfg(2, 32, 2);
        cudaLaunchKernelEx(&cfg, dsmem_lat_chained<2>, d_lat);
        cudaDeviceSynchronize();
        cudaLaunchKernelEx(&cfg, dsmem_lat_chained<2>, d_lat);
        cudaDeviceSynchronize();
        unsigned long long c; cudaMemcpy(&c, d_lat, 8, cudaMemcpyDeviceToHost);
        printf("  Cluster=2 DSMEM:            %.2f cy/load\n", (double)c / CHAIN_LEN);
    }
    {
        auto cfg = make_cfg(4, 32, 4);
        cudaLaunchKernelEx(&cfg, dsmem_lat_chained<4>, d_lat);
        cudaDeviceSynchronize();
        cudaLaunchKernelEx(&cfg, dsmem_lat_chained<4>, d_lat);
        cudaDeviceSynchronize();
        unsigned long long c; cudaMemcpy(&c, d_lat, 8, cudaMemcpyDeviceToHost);
        printf("  Cluster=4 DSMEM:            %.2f cy/load\n", (double)c / CHAIN_LEN);
    }
    {
        auto cfg = make_cfg(8, 32, 8);
        cudaLaunchKernelEx(&cfg, dsmem_lat_chained<8>, d_lat);
        cudaDeviceSynchronize();
        cudaLaunchKernelEx(&cfg, dsmem_lat_chained<8>, d_lat);
        cudaDeviceSynchronize();
        unsigned long long c; cudaMemcpy(&c, d_lat, 8, cudaMemcpyDeviceToHost);
        printf("  Cluster=8 DSMEM:            %.2f cy/load\n", (double)c / CHAIN_LEN);
    }

    // === BW: cluster_size sweep ===
    printf("\n--- DSMEM BW (varying offset, 8-way ILP) ---\n");

    int iters = 5000;
    int blocks_per_cluster_test = 144;  // ~full GPU

    {
        auto cfg = make_cfg(blocks_per_cluster_test, 128, 2);
        for (int i = 0; i < 3; i++) cudaLaunchKernelEx(&cfg, dsmem_bw_varying<2>, d_out, iters);
        cudaDeviceSynchronize();
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int i = 0; i < 5; i++) cudaLaunchKernelEx(&cfg, dsmem_bw_varying<2>, d_out, iters);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
        double bytes = (double)blocks_per_cluster_test * 128 * iters * 8 * 4;
        double tbs = bytes / 1e12 / (ms / 1000.0);
        printf("  Cluster=2 DSMEM BW: %.2f ms → %.2f TB/s\n", ms, tbs);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    {
        auto cfg = make_cfg(blocks_per_cluster_test, 128, 4);
        for (int i = 0; i < 3; i++) cudaLaunchKernelEx(&cfg, dsmem_bw_varying<4>, d_out, iters);
        cudaDeviceSynchronize();
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int i = 0; i < 5; i++) cudaLaunchKernelEx(&cfg, dsmem_bw_varying<4>, d_out, iters);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
        double bytes = (double)blocks_per_cluster_test * 128 * iters * 8 * 4;
        double tbs = bytes / 1e12 / (ms / 1000.0);
        printf("  Cluster=4 DSMEM BW: %.2f ms → %.2f TB/s\n", ms, tbs);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    {
        auto cfg = make_cfg(blocks_per_cluster_test, 128, 8);
        for (int i = 0; i < 3; i++) cudaLaunchKernelEx(&cfg, dsmem_bw_varying<8>, d_out, iters);
        cudaDeviceSynchronize();
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int i = 0; i < 5; i++) cudaLaunchKernelEx(&cfg, dsmem_bw_varying<8>, d_out, iters);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
        double bytes = (double)blocks_per_cluster_test * 128 * iters * 8 * 4;
        double tbs = bytes / 1e12 / (ms / 1000.0);
        printf("  Cluster=8 DSMEM BW: %.2f ms → %.2f TB/s\n", ms, tbs);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    cudaFree(d_lat);
    cudaFree(d_out);
    return 0;
}
