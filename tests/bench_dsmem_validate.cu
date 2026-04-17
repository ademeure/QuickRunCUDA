// DSMEM validation: write distinct pattern in each CTA, read remote, verify it matches
// the expected REMOTE CTA's data, NOT local data.

#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned smem[];

extern "C" __global__ __cluster_dims__(4, 1, 1) __launch_bounds__(128, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    // Get cluster-local CTA ID
    unsigned cluster_ctaid;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(cluster_ctaid));

    // Initialize my smem with cluster_ctaid as a marker
    unsigned my_marker = (cluster_ctaid << 24) | 0xABCDEF;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        smem[i] = my_marker | i;
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");

    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned target_cta = (cluster_ctaid + 1) % 4;
    unsigned remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote_addr) : "r"(smem_addr), "r"(target_cta));

    // Read remote smem[0]
    unsigned remote_val;
    asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(remote_val) : "r"(remote_addr));
    // Read local smem[0] for comparison
    unsigned local_val;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(local_val) : "r"(smem_addr));

    // Output: each thread of CTA 0 writes its (cluster_ctaid, target_cta, remote_val, local_val)
    if (threadIdx.x == 0) {
        unsigned* out = C + (blockIdx.x * 8);
        out[0] = cluster_ctaid;
        out[1] = target_cta;
        out[2] = remote_val;
        out[3] = local_val;
        out[4] = my_marker;  // what I wrote
        // Expected remote val = ((target_cta) << 24) | 0xABCDEF | 0
        out[5] = (target_cta << 24) | 0xABCDEF;  // expected
    }
}
