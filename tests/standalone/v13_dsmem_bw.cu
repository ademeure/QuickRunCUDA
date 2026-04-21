// V13: DCE-immune DSMEM throughput measurement
// Methodology: N parallel dependent chains → each chain is DCE-immune,
//  but N chains run in parallel revealing effective throughput.
//
// - 32 threads per CTA each with own dep-chain (32× parallel chains)
// - Optional per-thread ILP sweep: 1/2/4 chains per thread
// - clock64 boundaries → DCE-immune timing
// - cluster_dims(CX) with peer = (my_cta+1) % CX
// - CL short to avoid crash at cluster≥3

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256

// 32 threads × 1 chain each — pure parallelism
template<int CX, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void dsmem_bw_1(unsigned long long* out, unsigned seed) {
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

    // Each thread starts at different initial offset
    unsigned cur = (tid * 4u) & (SMEM_W*4 - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur) : "r"(peer_base + cur) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
    }
    if (my_cta == 0 && blockIdx.x == 0) {
        ((unsigned*)out)[2 + tid] = cur;  // anti-DCE per thread
    }
}

// 32 threads × ILP=4 chains each — 128 parallel chains total
template<int CX, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void dsmem_bw_4(unsigned long long* out, unsigned seed) {
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

    // 4 independent starting points per thread
    unsigned c0 = (tid * 4u + 0u)   & (SMEM_W*4 - 1);
    unsigned c1 = (tid * 4u + 64u)  & (SMEM_W*4 - 1);
    unsigned c2 = (tid * 4u + 128u) & (SMEM_W*4 - 1);
    unsigned c3 = (tid * 4u + 192u) & (SMEM_W*4 - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(c0) : "r"(peer_base + c0) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(c1) : "r"(peer_base + c1) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(c2) : "r"(peer_base + c2) : "memory");
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(c3) : "r"(peer_base + c3) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
    }
    if (my_cta == 0 && blockIdx.x == 0) {
        ((unsigned*)out)[2 + tid] = c0 ^ c1 ^ c2 ^ c3;  // anti-DCE
    }
}

// Local SMEM 32-thread baseline
template<int CL>
__global__ __launch_bounds__(32, 1)
void local_bw_1(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur = (tid * 4u) & (SMEM_W*4 - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(cur) : "r"(local_base + cur) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) out[0] = t1 - t0;
    if (blockIdx.x == 0) ((unsigned*)out)[2 + tid] = cur;
}

template<int CX, int CL, int ILP, typename K>
static int run_dsmem_bw(K kernel, unsigned long long* d_out, double* ns_out, double* cy_out, int N, double clock_ghz) {
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

    int got = 0, crashes = 0;
    for (int r = 0; r < N*2 && got < N; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, d_out, 42u + r);
        if (e) { crashes++; continue; }
        e = cudaDeviceSynchronize();
        if (e) { crashes++; continue; }
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        cy_out[got] = (double)cy;
        ns_out[got] = (double)cy / clock_ghz;
        got++;
    }
    if (crashes > 0) fprintf(stderr, "  (cx=%d cl=%d ilp=%d had %d crashes)\n", CX, CL, ILP, crashes);
    return (got == N) ? 0 : -1;
}

static void stats(const double* s, int n, double& mn, double& avg, double& mx) {
    mn = 1e30; mx = 0; avg = 0;
    for (int i = 0; i < n; i++) { if (s[i]<mn) mn=s[i]; if (s[i]>mx) mx=s[i]; avg+=s[i]; }
    avg /= n;
}

int main() {
    CK(cudaSetDevice(0));
    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    const double CLOCK_GHZ = 1.920;
    const int N_SAMPLES = 30;
    double cy[64], ns[64];
    double mn, avg, mx;

    printf("=== V13 DSMEM BW — DCE-immune parallel dep-chains (1920 MHz locked) ===\n\n");

    // Local baseline (32 threads × ILP=1 chains)
    if (run_dsmem_bw<0, 100, 1>(local_bw_1<100>, d_out, ns, cy, N_SAMPLES, CLOCK_GHZ) == 0) {
        stats(cy, N_SAMPLES, mn, avg, mx);
        double loads = 32.0 * 100.0;  // 32 threads × 100 CL
        double bytes = loads * 4.0;
        double ghz_avg_cy = avg;
        double bw_tbs = bytes / (ghz_avg_cy / CLOCK_GHZ / 1e9) / 1e12;
        double cy_per_load = avg / 100.0;  // per thread per iter
        printf("LOCAL 32t ILP=1 CL=100:  %8.0f cy total (%5.2f cy/load/thread, min %5.2f, max %5.2f), BW=%.2f TB/s\n",
               avg, cy_per_load, mn/100.0, mx/100.0, bw_tbs);
    }

    printf("\n--- DSMEM BW: 32 threads × ILP=1 (CL=5) ---\n");
    for (int cx : {2, 4, 8}) {
        int rc = -1;
        if (cx == 2) rc = run_dsmem_bw<2, 5, 1>(dsmem_bw_1<2, 5>, d_out, ns, cy, N_SAMPLES, CLOCK_GHZ);
        if (cx == 4) rc = run_dsmem_bw<4, 5, 1>(dsmem_bw_1<4, 5>, d_out, ns, cy, N_SAMPLES, CLOCK_GHZ);
        if (cx == 8) rc = run_dsmem_bw<8, 5, 1>(dsmem_bw_1<8, 5>, d_out, ns, cy, N_SAMPLES, CLOCK_GHZ);
        if (rc == 0) {
            stats(cy, N_SAMPLES, mn, avg, mx);
            double loads = 32.0 * 5.0;
            double bytes = loads * 4.0;
            double bw_tbs = bytes / (avg / CLOCK_GHZ / 1e9) / 1e12;
            double cy_per_load = avg / 5.0;
            printf("DSMEM cx=%d 32t ILP=1 CL=5:  %8.0f cy total (%5.2f cy/load/thread, min %5.2f, max %5.2f), BW=%.2f TB/s\n",
                   cx, avg, cy_per_load, mn/5.0, mx/5.0, bw_tbs);
        }
    }

    printf("\n--- DSMEM BW: 32 threads × ILP=4 (CL=5, 128 parallel chains) ---\n");
    for (int cx : {2, 4, 8}) {
        int rc = -1;
        if (cx == 2) rc = run_dsmem_bw<2, 5, 4>(dsmem_bw_4<2, 5>, d_out, ns, cy, N_SAMPLES, CLOCK_GHZ);
        if (cx == 4) rc = run_dsmem_bw<4, 5, 4>(dsmem_bw_4<4, 5>, d_out, ns, cy, N_SAMPLES, CLOCK_GHZ);
        if (cx == 8) rc = run_dsmem_bw<8, 5, 4>(dsmem_bw_4<8, 5>, d_out, ns, cy, N_SAMPLES, CLOCK_GHZ);
        if (rc == 0) {
            stats(cy, N_SAMPLES, mn, avg, mx);
            double loads = 32.0 * 4.0 * 5.0;
            double bytes = loads * 4.0;
            double bw_tbs = bytes / (avg / CLOCK_GHZ / 1e9) / 1e12;
            double cy_per_load = avg / 5.0 / 4.0;
            printf("DSMEM cx=%d 32t ILP=4 CL=5:  %8.0f cy total (%5.2f cy/load/thread/chain, min %5.2f, max %5.2f), BW=%.2f TB/s\n",
                   cx, avg, cy_per_load, mn/20.0, mx/20.0, bw_tbs);
        }
    }

    cudaFree(d_out);
    return 0;
}
