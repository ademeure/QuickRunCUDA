// V27: DSMEM + mbarrier async arrival — producer-consumer pattern
// Compare: 1) fence.sc.cluster + barrier.cluster
//          2) mbarrier in peer CTA with arrive/wait
//
// Goal: find fastest cluster-scoped producer-consumer mechanism.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 8

// A) Producer/consumer via barrier.cluster.arrive/wait (legacy)
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void pc_barrier(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) smem[i] = 0;
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(1u));

    unsigned addr_peer = peer_base + (seed & 0xFC);
    unsigned addr_local = local_base + (seed & 0xFC);
    unsigned val = 0;

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (my_cta == 0 && tid == 0) {
        // Producer loop: write to peer, signal via barrier
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("st.shared::cluster.u32 [%0], %1;"
                         :: "r"(addr_peer), "r"(seed + i) : "memory");
            asm volatile("barrier.cluster.arrive;" ::: "memory");
            asm volatile("barrier.cluster.wait;"  ::: "memory");
        }
    } else if (my_cta == 1 && tid == 0) {
        // Consumer loop
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("barrier.cluster.arrive;" ::: "memory");
            asm volatile("barrier.cluster.wait;"  ::: "memory");
            asm volatile("ld.shared.u32 %0, [%1];"
                         : "=r"(val) : "r"(addr_local) : "memory");
        }
    } else {
        // Others: participate in barrier
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("barrier.cluster.arrive;" ::: "memory");
            asm volatile("barrier.cluster.wait;"  ::: "memory");
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 1 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = val;
    }
}

// B) Ring write with batch fence (less overhead)
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void pc_batch_fence(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) smem[i] = 0;
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(1u));

    unsigned addr_peer = peer_base + (seed & 0xFC);
    unsigned addr_local = local_base + (seed & 0xFC);
    unsigned val = 0;

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Producer batched: CL writes then 1 fence + 1 barrier
    if (my_cta == 0 && tid == 0) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("st.shared::cluster.u32 [%0], %1;"
                         :: "r"(addr_peer), "r"(seed + i) : "memory");
        }
        asm volatile("fence.sc.cluster;" ::: "memory");
        asm volatile("barrier.cluster.arrive;" ::: "memory");
        asm volatile("barrier.cluster.wait;"  ::: "memory");
    } else if (my_cta == 1 && tid == 0) {
        asm volatile("barrier.cluster.arrive;" ::: "memory");
        asm volatile("barrier.cluster.wait;"  ::: "memory");
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(val) : "r"(addr_local) : "memory");
    } else {
        asm volatile("barrier.cluster.arrive;" ::: "memory");
        asm volatile("barrier.cluster.wait;"  ::: "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 1 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = val;
    }
}

// C) mbarrier-based producer-consumer
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void pc_mbarrier(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    __shared__ unsigned long long mbar;

    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) smem[i] = 0;
    if (tid == 0) {
        // mbarrier init
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned mbar_local = (unsigned)__cvta_generic_to_shared(&mbar);
    unsigned peer_mbar_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_mbar_base) : "r"(mbar_local), "r"(1u));
    unsigned peer_smem_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_smem_base) : "r"(local_base), "r"(1u));

    unsigned addr_peer = peer_smem_base + (seed & 0xFC);
    unsigned addr_local = local_base + (seed & 0xFC);
    unsigned val = 0;
    unsigned long long token;

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (my_cta == 0 && tid == 0) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("st.shared::cluster.u32 [%0], %1;"
                         :: "r"(addr_peer), "r"(seed + i) : "memory");
            // Signal peer's mbarrier (cluster variant: no token dest)
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         :: "r"(peer_mbar_base) : "memory");
        }
    } else if (my_cta == 1 && tid == 0) {
        unsigned long long phase = 0;
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            // Wait on local mbarrier using try_wait (phase alternates)
            int done = 0;
            int spin_guard = 0;
            while (!done && spin_guard < 100000) {
                asm volatile("{.reg .pred p;\n"
                             "mbarrier.try_wait.shared.b64 p, [%1], %2;\n"
                             "selp.u32 %0, 1, 0, p;}\n"
                             : "=r"(done) : "r"(mbar_local), "l"(phase) : "memory");
                spin_guard++;
            }
            phase = 1 - phase;  // flip phase for next arrival
            asm volatile("ld.shared.u32 %0, [%1];"
                         : "=r"(val) : "r"(addr_local) : "memory");
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 1 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = val;
    }
}

template<typename K, typename... Args>
static int avg_cy_t(int threads, K kernel, int N, double* cy_out, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CX, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
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
        cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, d_out, args..., 42u + r);
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

    printf("=== V27 DSMEM producer-consumer patterns (1920 MHz) ===\n\n");

    // A) Barrier per message
    printf("A) barrier.cluster per message (CL=20 handoffs):\n");
    int got = avg_cy_t(32, pc_barrier<20>, 15, &cy);
    if (got > 0) {
        double t_us = cy / CLOCK_GHZ / 1e3;
        printf("  Total: %.0f cy (%.2f us), per msg: %.0f cy (%.2f us)\n",
               cy, t_us, cy/20, t_us/20);
    }

    // B) Batched fence (1 fence for CL writes)
    printf("\nB) Batched fence for CL=20 writes then 1 handoff:\n");
    got = avg_cy_t(32, pc_batch_fence<20>, 15, &cy);
    if (got > 0) {
        double t_us = cy / CLOCK_GHZ / 1e3;
        printf("  Total: %.0f cy (%.2f us), per msg amortized: %.0f cy (%.3f us)\n",
               cy, t_us, cy/20, t_us/20);
    }

    // C) mbarrier
    printf("\nC) mbarrier.shared::cluster (CL=20 handoffs):\n");
    got = avg_cy_t(32, pc_mbarrier<20>, 15, &cy);
    if (got > 0) {
        double t_us = cy / CLOCK_GHZ / 1e3;
        printf("  Total: %.0f cy (%.2f us), per msg: %.0f cy (%.2f us)\n",
               cy, t_us, cy/20, t_us/20);
    } else printf("  mbarrier test FAILED\n");

    return 0;
}
