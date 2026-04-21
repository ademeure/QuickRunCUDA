// V7 D1: Cluster-wide all-reduce primitive via DSMEM + cluster.barrier
// Each CTA contributes value; all CTAs see sum at end
// MODE 0: 4 CTA cluster, sum 4 floats
// MODE 1: 8 CTA cluster, sum 8 floats
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
#define CSIZE 4
#elif MODE == 1
#define CSIZE 8
#endif

extern "C" __global__ __launch_bounds__(128, 1)
__cluster_dims__(CSIZE, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ float local_val;
    if (threadIdx.x == 0) local_val = (float)blockIdx.x;

    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    float final_sum = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // All CTAs reduce into final_sum by reading peer DSMEM
        unsigned int local_addr = __cvta_generic_to_shared(&local_val);
        float sum = 0;
        #pragma unroll
        for (int peer = 0; peer < CSIZE; peer++) {
            unsigned int peer_addr;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                         : "=r"(peer_addr) : "r"(local_addr), "r"(peer));
            float v;
            asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(peer_addr));
            sum += v;
        }
        // Sync before next iter
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
        final_sum = sum;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (final_sum == 1.234567e-30f) C[blockIdx.x] = final_sum;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("CSIZE=%d final_sum=%.0f cy/iter=%.3f\n",
               CSIZE, final_sum, (double)(t1-t0)/(double)ITERS);
    }
}
