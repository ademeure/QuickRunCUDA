// V17: Many-to-many DSMEM — does the interconnect share bandwidth?
//
// Tests:
// A) 1 CTA reading 1 peer (baseline) — V15 style, single thread
// B) Ring: N CTAs each reading its +1 neighbor — 1 thread per CTA
// C) All-to-all: every CTA reads from every other CTA in a rotating sequence
// D) Same tests at ILP=4
//
// Key question: does each CTA see N× slowdown when N CTAs all reading simultaneously?
// If yes → shared bus.  If no → point-to-point.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 8
#define CL 5

// Baseline: only SRC=0 reads from DST, others idle (spin wait)
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void one_to_one(unsigned long long* out, unsigned dst, unsigned seed) {
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
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(dst));

    unsigned long long t0 = 0, t1 = 0;
    unsigned cur = 0;

    if (my_cta == 0 && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(peer_base + cur) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

// Ring: all CTAs read from (my+1) % CX
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void many_ring(unsigned long long* out, unsigned seed) {
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
    unsigned target_cta = (my_cta + 1u) % CX;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    unsigned cur = 0;

    // All CTAs sync right before the timed region for simultaneity
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (tid == 0) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(peer_base + cur) : "memory");
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

// Two-sender variant: only CTAs 0 and 4 do loads, others idle — verify pair contention
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void two_senders(unsigned long long* out, unsigned dst1, unsigned seed) {
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
    unsigned cur = 0;

    // CTA 0 reads from 1, CTA 4 reads from dst1 (configurable)
    unsigned peer_base = 0;
    bool active = false;
    unsigned target = 0;
    if (my_cta == 0) { target = 1; active = true; }
    if (my_cta == 4) { target = dst1; active = true; }
    if (active) {
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                     : "=r"(peer_base) : "r"(local_base), "r"(target));
    }

    // Sync right before timing
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (active && tid == 0) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(peer_base + cur) : "memory");
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

// N senders all reading neighbor (ring pattern but variable N)
// Only first N CTAs actively read; rest idle.
template<int N_ACTIVE>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void n_senders(unsigned long long* out, unsigned seed) {
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
    unsigned target_cta = (my_cta + 1u) % CX;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    unsigned cur = 0;
    bool active = (my_cta < N_ACTIVE);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (active && tid == 0) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(peer_base + cur) : "memory");
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

template<typename K, typename... Args>
static int avg_runs(K kernel, int N, double* cy_out, Args... args) {
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
    double cy;

    printf("=== V17 DSMEM many-to-many contention (CX=8, 1 thread per CTA, CL=5) ===\n");
    printf("(SMs: CTA0=SM0, CTA1=SM1, CTA2=SM16, CTA3=SM17, CTA4=SM32, CTA5=SM33, CTA6=SM48, CTA7=SM49)\n\n");

    // A) Baseline: CTA 0 reads from various single peers
    printf("A) Single CTA 0 reading single peer:\n");
    for (int d = 1; d < CX; d++) {
        int n = avg_runs(one_to_one, 20, &cy, (unsigned)d);
        if (n > 0) printf("  CTA0→CTA%d (SM0→SM%d): %.2f cy/load\n",
                          d, d==1?1 : d==2?16 : d==3?17 : d==4?32 : d==5?33 : d==6?48 : 49, cy/CL);
    }

    // B) N senders (1..8), each reading (my+1)%CX neighbor
    printf("\nB) N senders active (each reads (my+1)%%8 — SM0 is CTA 0, always timing):\n");
    {
        double cy_single;
        int n = avg_runs(n_senders<1>, 20, &cy_single);
        if (n > 0) printf("  N=1 active:  %.2f cy/load (reference)\n", cy_single/CL);

        n = avg_runs(n_senders<2>, 20, &cy);
        if (n > 0) printf("  N=2 active:  %.2f cy/load  (%.2fx vs N=1)\n", cy/CL, cy/cy_single);
        n = avg_runs(n_senders<3>, 20, &cy);
        if (n > 0) printf("  N=3 active:  %.2f cy/load  (%.2fx)\n", cy/CL, cy/cy_single);
        n = avg_runs(n_senders<4>, 20, &cy);
        if (n > 0) printf("  N=4 active:  %.2f cy/load  (%.2fx)\n", cy/CL, cy/cy_single);
        n = avg_runs(n_senders<5>, 20, &cy);
        if (n > 0) printf("  N=5 active:  %.2f cy/load  (%.2fx)\n", cy/CL, cy/cy_single);
        n = avg_runs(n_senders<6>, 20, &cy);
        if (n > 0) printf("  N=6 active:  %.2f cy/load  (%.2fx)\n", cy/CL, cy/cy_single);
        n = avg_runs(n_senders<7>, 20, &cy);
        if (n > 0) printf("  N=7 active:  %.2f cy/load  (%.2fx)\n", cy/CL, cy/cy_single);
        n = avg_runs(n_senders<8>, 20, &cy);
        if (n > 0) printf("  N=8 active:  %.2f cy/load  (%.2fx)\n", cy/CL, cy/cy_single);
    }

    // C) Two senders going to different DSTs (contention isolation)
    printf("\nC) Two senders (CTA 0 → CTA 1, CTA 4 → CTA X) — tests pairwise contention:\n");
    for (int d : {2, 3, 5, 6, 7}) {
        int n = avg_runs(two_senders, 20, &cy, (unsigned)d);
        if (n > 0) printf("  CTA0→CTA1 + CTA4→CTA%d (SM32→...): %.2f cy/load on CTA 0\n", d, cy/CL);
    }

    return 0;
}
