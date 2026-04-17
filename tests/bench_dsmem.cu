// DSMEM (Distributed Shared Memory) bandwidth test.
// Uses CGA cluster_dim to enable cross-CTA smem access.
// Each CTA reads from another CTA's smem via cluster.

#ifndef OP
#define OP 0
#endif
#ifndef CLUSTER_X
#define CLUSTER_X 4
#endif

extern __shared__ unsigned smem[];

extern "C" __global__ __cluster_dims__(CLUSTER_X, 1, 1) __launch_bounds__(128, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    // Init my smem with a pattern
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        smem[i] = (blockIdx.x << 16) | i;
    }
    __syncthreads();
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");

    unsigned acc = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Get neighbor CTA's smem address. Use cluster shuffle.
    unsigned my_cta = blockIdx.x % CLUSTER_X;
    unsigned target_cta = (my_cta + 1) % CLUSTER_X;

    // mapa: map my smem addr to target CTA's smem
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote_addr) : "r"(smem_addr), "r"(target_cta));

#if OP == 0
    // Read remote smem 1024 times
    #pragma unroll 1
    for (int i = 0; i < 1024; i++) {
        unsigned r;
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r) : "r"(remote_addr + (threadIdx.x * 4)));
        acc ^= r;
    }
#elif OP == 1
    // Local smem access for comparison
    #pragma unroll 1
    for (int i = 0; i < 1024; i++) {
        unsigned r;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(smem_addr + (threadIdx.x * 4)));
        acc ^= r;
    }
#endif

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
    }
}
