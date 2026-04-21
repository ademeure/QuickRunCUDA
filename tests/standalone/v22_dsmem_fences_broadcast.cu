// V22: DSMEM fence cost comparison + broadcast pattern
// A) Fence scope: fence.acq_rel.cluster / fence.sc.cluster / fence.sc.gpu / fence.sc.sys
// B) Broadcast: CTA 0 writes, CTA 1..7 read via DSMEM (producer→consumers)
// C) DSMEM atomic scope: .cta / .cluster / .gpu variants

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 8

enum FenceKind { F_NONE = 0, F_ACQ_REL_CLUSTER = 1, F_SC_CLUSTER = 2, F_SC_GPU = 3, F_SC_SYS = 4 };

template<int FK, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void fence_variants(unsigned long long* out, unsigned seed) {
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

    unsigned addr = peer_base + (seed & (SMEM_W*4-4));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    if (my_cta == 0 && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("st.shared::cluster.u32 [%0], %1;"
                         :: "r"(addr), "r"(seed + i) : "memory");
            if (FK == F_ACQ_REL_CLUSTER) asm volatile("fence.acq_rel.cluster;" ::: "memory");
            else if (FK == F_SC_CLUSTER) asm volatile("fence.sc.cluster;" ::: "memory");
            else if (FK == F_SC_GPU) asm volatile("fence.sc.gpu;" ::: "memory");
            else if (FK == F_SC_SYS) asm volatile("fence.sc.sys;" ::: "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

// Broadcast: CTA 0 writes chunk, CTAs 1..CX-1 read same data simultaneously
template<int ILP, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void broadcast(unsigned long long* out, unsigned seed) {
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
    unsigned cta0_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(cta0_base) : "r"(local_base), "r"(0u));

    unsigned c[8];
    #pragma unroll
    for (int k = 0; k < ILP; k++) c[k] = (tid * 4u + k * 32u) & (SMEM_W*4 - 1);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // CTAs 1..CX-1 read from CTA 0's SMEM
    if (my_cta != 0) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                asm volatile("ld.shared::cluster.u32 %0, [%1];"
                             : "=r"(c[k]) : "r"(cta0_base + c[k]) : "memory");
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    // CTA 1 reports timing (not CTA 0 which was idle)
    if (my_cta == 1 && tid == 0) {
        out[0] = t1 - t0;
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= c[k];
        ((unsigned*)out)[2] = acc;
    }
}

// Atomic scope sweep
enum AtomScope { A_CTA = 0, A_CLUSTER = 1, A_GPU = 2 };

template<int AS, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void atom_scope(unsigned long long* out, unsigned seed) {
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

    unsigned addr = peer_base + (seed & (SMEM_W*4-4));
    unsigned acc = 0;
    unsigned long long t0 = 0, t1 = 0;

    if (my_cta == 0 && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            if (AS == A_CTA)
                asm volatile("atom.add.shared.u32 %0, [%1], %2;"
                             : "=r"(acc) : "r"(addr), "r"(acc + i) : "memory");
            else if (AS == A_CLUSTER)
                asm volatile("atom.add.shared::cluster.u32 %0, [%1], %2;"
                             : "=r"(acc) : "r"(addr), "r"(acc + i) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = acc;
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

    printf("=== V22 DSMEM fence scopes + broadcast + atom scopes ===\n\n");

    // A) Fence scopes
    printf("A) Fence scope cost (1 thread, CL=20 writes, each followed by fence):\n");
    double cy_none;
    avg_cy_t(32, fence_variants<F_NONE, 20>, 20, &cy_none);
    printf("  no fence:              %.0f cy (%.2f cy/st)\n", cy_none, cy_none/20);

    double cy_acq;
    avg_cy_t(32, fence_variants<F_ACQ_REL_CLUSTER, 20>, 20, &cy_acq);
    printf("  fence.acq_rel.cluster: %.0f cy (%.2f cy/(st+fence), +%.1f cy/fence)\n",
           cy_acq, cy_acq/20, (cy_acq - cy_none)/20);

    double cy_sc_cl;
    avg_cy_t(32, fence_variants<F_SC_CLUSTER, 20>, 20, &cy_sc_cl);
    printf("  fence.sc.cluster:      %.0f cy (%.2f cy/(st+fence), +%.1f cy/fence)\n",
           cy_sc_cl, cy_sc_cl/20, (cy_sc_cl - cy_none)/20);

    double cy_sc_gpu;
    avg_cy_t(32, fence_variants<F_SC_GPU, 20>, 20, &cy_sc_gpu);
    printf("  fence.sc.gpu:          %.0f cy (%.2f cy/(st+fence), +%.1f cy/fence)\n",
           cy_sc_gpu, cy_sc_gpu/20, (cy_sc_gpu - cy_none)/20);

    double cy_sc_sys;
    avg_cy_t(32, fence_variants<F_SC_SYS, 20>, 20, &cy_sc_sys);
    printf("  fence.sc.sys:          %.0f cy (%.2f cy/(st+fence), +%.1f cy/fence)\n",
           cy_sc_sys, cy_sc_sys/20, (cy_sc_sys - cy_none)/20);

    // B) Broadcast
    printf("\nB) Broadcast (CTA 0 source, CTAs 1..7 all read CTA 0's SMEM):\n");
    printf("  ILP=1 CL=5: ");
    int got = avg_cy_t(32, broadcast<1, 5>, 15, &cy);
    if (got > 0) {
        double loads_per_cta = 32.0 * 1 * 5;
        double total_loads = loads_per_cta * (CX-1);
        double bytes = total_loads * 4.0;
        double t = cy / CLOCK_GHZ / 1e9;
        printf("%.0f cy, cy/load per CTA=%.2f, BW_total=%.2f GB/s\n",
               cy, cy/loads_per_cta, bytes/t/1e9);
    }
    printf("  ILP=4 CL=5: ");
    got = avg_cy_t(32, broadcast<4, 5>, 15, &cy);
    if (got > 0) {
        double loads_per_cta = 32.0 * 4 * 5;
        double total_loads = loads_per_cta * (CX-1);
        double bytes = total_loads * 4.0;
        double t = cy / CLOCK_GHZ / 1e9;
        printf("%.0f cy, cy/load per CTA=%.2f, BW_total=%.2f GB/s\n",
               cy, cy/loads_per_cta, bytes/t/1e9);
    }
    printf("  ILP=8 CL=5: ");
    got = avg_cy_t(32, broadcast<8, 5>, 15, &cy);
    if (got > 0) {
        double loads_per_cta = 32.0 * 8 * 5;
        double total_loads = loads_per_cta * (CX-1);
        double bytes = total_loads * 4.0;
        double t = cy / CLOCK_GHZ / 1e9;
        printf("%.0f cy, cy/load per CTA=%.2f, BW_total=%.2f GB/s\n",
               cy, cy/loads_per_cta, bytes/t/1e9);
    }

    // C) Atom scope
    printf("\nC) Atomic scope (CTA 0 atom.add to CTA 1, CL=20):\n");
    got = avg_cy_t(32, atom_scope<A_CTA, 20>, 20, &cy);
    if (got > 0) printf("  atom.add.shared.u32            (.cta):     %.0f cy (%.2f cy/atom)\n", cy, cy/20);
    got = avg_cy_t(32, atom_scope<A_CLUSTER, 20>, 20, &cy);
    if (got > 0) printf("  atom.add.shared::cluster.u32   (.cluster): %.0f cy (%.2f cy/atom)\n", cy, cy/20);

    return 0;
}
