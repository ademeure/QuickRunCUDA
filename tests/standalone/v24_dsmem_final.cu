// V24: Final DSMEM exploration — isolated tests
// A) Local atomic scope: .cta vs .cluster on LOCAL smem (no cluster launch)
// B) DSMEM + local SMEM concurrent (separate clock64 per group)
// C) Writes under load: does hot-spot write contention exist?
// D) Bank conflicts on local smem baseline (control)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 1024   // wider to avoid overflow
#define CX 8

// ============ A) Local atomic scope (no cluster) ============
template<int USE_CLUSTER_SCOPE, int CL>
__global__ __launch_bounds__(32, 1)
void atom_local(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) smem[i] = 0;
    __syncthreads();
    if (tid != 0) return;

    unsigned addr = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned acc = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        if (USE_CLUSTER_SCOPE)
            asm volatile("atom.add.shared::cluster.u32 %0, [%1], %2;"
                         : "=r"(acc) : "r"(addr), "r"(acc + i) : "memory");
        else
            asm volatile("atom.add.shared.u32 %0, [%1], %2;"
                         : "=r"(acc) : "r"(addr), "r"(acc + i) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    ((unsigned*)out)[2] = acc;
}

// Local atomic scopes available: .cta (default) and .gpu on local SMEM address
template<int CL>
__global__ __launch_bounds__(32, 1)
void atom_local_gpu(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) smem[i] = 0;
    __syncthreads();
    if (tid != 0) return;

    unsigned addr = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned acc = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("atom.add.gpu.shared.u32 %0, [%1], %2;"
                     : "=r"(acc) : "r"(addr), "r"(acc + i) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    ((unsigned*)out)[2] = acc;
}

// ============ B) DSMEM + local SMEM concurrent (clock each half) ============
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(64, 1)
void mixed_read(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 64) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"((my_cta + 1u) % CX));

    int warp_id = tid / 32;
    bool is_local_warp = (warp_id == 0);  // warp 0: local, warp 1: DSMEM

    unsigned cur = (tid * 4u) & (SMEM_W*4 - 1);
    unsigned base = is_local_warp ? local_base : peer_base;

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (is_local_warp) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared.u32 %0, [%1];"
                         : "=r"(cur) : "r"(base + cur) : "memory");
        }
    } else {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(base + cur) : "memory");
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;  // local warp time
        ((unsigned*)out)[4] = cur;
    }
    if (my_cta == 0 && tid == 32) {
        out[1] = t1 - t0;  // DSMEM warp time
        ((unsigned*)out)[5] = cur;
    }
}

// LOCAL-only baseline (1 warp, chain)
template<int CL>
__global__ __launch_bounds__(32, 1)
void local_only(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();

    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur = (tid * 4u) & (SMEM_W*4 - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(cur) : "r"(base + cur) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) out[0] = t1 - t0;
    ((unsigned*)out)[2] = cur;
}

// ============ C) Hot-spot writes ============
template<int N_ACTIVE, int ILP, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void hot_write(unsigned long long* out, unsigned seed) {
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
                 : "=r"(peer_base) : "r"(local_base), "r"(0u));  // all target CTA 0

    unsigned a[8];
    #pragma unroll
    for (int k = 0; k < ILP; k++) a[k] = peer_base + ((tid * 4u + k * 32u) % (SMEM_W*4 - 16));
    unsigned v = seed + tid;
    bool active = (my_cta < N_ACTIVE && my_cta != 0);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    if (active) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                asm volatile("st.shared::cluster.u32 [%0], %1;"
                             :: "r"(a[k]), "r"(v + i + k) : "memory");
            }
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    // Reporter: CTA 1 (first active writer)
    if (my_cta == 1 && tid == 0) out[0] = t1 - t0;
}

template<typename K, typename... Args>
static int avg_cy_t(int threads, K kernel, int N, double* cy_out, int cluster_launch, Args... args) {
    unsigned long long* d_out;
    cudaMalloc(&d_out, 32);
    double sum = 0; int got = 0;
    for (int r = 0; r < N * 2 && got < N; r++) {
        cudaError_t e;
        if (cluster_launch) {
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
            e = cudaLaunchKernelEx(&cfg, kernel, d_out, args..., 42u + r);
        } else {
            kernel<<<1, threads>>>(d_out, args..., 42u + r);
            e = cudaGetLastError();
        }
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

    printf("=== V24 DSMEM final corners ===\n\n");

    // A) Local atomic scope
    printf("A) Local SMEM atomic scope comparison (single thread, CL=100):\n");
    int got = avg_cy_t(32, atom_local<0, 100>, 20, &cy, 0);
    if (got > 0) printf("  atom.add.shared.u32            (default .cta): %.0f cy (%.2f cy/atom)\n", cy, cy/100);
    got = avg_cy_t(32, atom_local<1, 100>, 20, &cy, 0);
    if (got > 0) printf("  atom.add.shared::cluster.u32   (.cluster):      %.0f cy (%.2f cy/atom)\n", cy, cy/100);
    got = avg_cy_t(32, atom_local_gpu<100>, 20, &cy, 0);
    if (got > 0) printf("  atom.add.gpu.shared.u32        (.gpu):          %.0f cy (%.2f cy/atom)\n", cy, cy/100);

    // B) Concurrent DSMEM + local
    printf("\nB) Concurrent DSMEM + local (2 warps: warp 0 local, warp 1 DSMEM):\n");
    unsigned long long* d_out_alloc;
    cudaMalloc(&d_out_alloc, 32);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CX, 1, 1);
    cfg.blockDim = dim3(64, 1, 1);
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CX;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    double sum_local = 0, sum_dsmem = 0; int n = 0;
    for (int r = 0; r < 15; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, mixed_read<50>, d_out_alloc, 42u + r);
        if (e) { cudaGetLastError(); continue; }
        e = cudaDeviceSynchronize();
        if (e) { cudaGetLastError(); continue; }
        unsigned long long vals[2];
        cudaMemcpy(vals, d_out_alloc, 16, cudaMemcpyDeviceToHost);
        sum_local += (double)vals[0];
        sum_dsmem += (double)vals[1];
        n++;
    }
    if (n > 0) {
        printf("  Local warp chain: %.0f cy (%.2f cy/load)\n", sum_local/n, sum_local/n/50);
        printf("  DSMEM warp chain: %.0f cy (%.2f cy/load)\n", sum_dsmem/n, sum_dsmem/n/50);

        // Compare local-only
        got = avg_cy_t(32, local_only<50>, 20, &cy, 0);
        if (got > 0) printf("  Local-only baseline: %.0f cy (%.2f cy/load) — delta vs concurrent: %+.1f cy\n",
                            cy, cy/50, sum_local/n - cy);
    }
    cudaFree(d_out_alloc);

    // C) Hot-spot writes
    printf("\nC) Hot-spot WRITES (all N senders write CTA 0, ILP=4):\n");
    printf("  N_active  cy_per_CTA  cy/store  BW_agg(GB/s)   Slowdown\n");
    double cy_ref = 0;
    #define HOTWR(N) { \
        int got = avg_cy_t(32, hot_write<N, 4, 5>, 20, &cy, 1); \
        if (got > 0) { \
            double stores_per = 32.0 * 4.0 * 5; \
            double t = cy / CLOCK_GHZ / 1e9; \
            double bw = stores_per * (N-1) * 4.0 / t / 1e9; \
            if (N == 2) cy_ref = cy; \
            printf("  %d         %8.0f   %5.2f     %8.2f       %.2fx\n", \
                   N, cy, cy/stores_per, bw, cy/cy_ref); \
        } }
    HOTWR(2); HOTWR(3); HOTWR(4); HOTWR(5); HOTWR(6); HOTWR(7); HOTWR(8);
    #undef HOTWR

    return 0;
}
