// V7 D2: Cluster broadcast pattern
// CTA 0 produces value, all other CTAs read it via DSMEM
// Compare to global memory broadcast and cluster.barrier alone
#ifndef MODE
#define MODE 0
#endif

#define CSIZE 4

extern "C" __global__ __launch_bounds__(128, 1)
__cluster_dims__(CSIZE, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ float val;
    if (blockIdx.x == 0 && threadIdx.x == 0) val = 1.0f;
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    float sum = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // CTA 0 writes; all CTAs read via DSMEM
        if (blockIdx.x == 0 && threadIdx.x == 0) val = (float)i;
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
        // All CTAs read CTA 0's val
        unsigned int local_addr = __cvta_generic_to_shared(&val);
        unsigned int peer_addr;
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_addr) : "r"(local_addr));
        float v;
        asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(peer_addr));
        sum += v;
#elif MODE == 1
        // CTA 0 writes to global, all CTAs read
        if (blockIdx.x == 0 && threadIdx.x == 0) A[0] = (float)i;
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
        sum += A[0];
#endif
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (sum == 1.234567e-30f) C[blockIdx.x] = sum;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d cy/iter=%.3f\n", MODE, (double)(t1-t0)/(double)ITERS);
    }
}
