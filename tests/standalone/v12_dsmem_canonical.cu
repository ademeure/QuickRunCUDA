// V12: DSMEM canonical latency — per-cluster CHAIN_LEN + N-sample statistics
// Fixes V11 crashes at cluster=4/8. Methodology matches 04_dsmem exactly.
//
// Crash-safe chain lengths (per 04_dsmem):
//   cluster=2: 50 iters (safe)
//   cluster=4/8: 5 iters (safe; 10 iters crashes ~50%, 15+ always)
//
// Local baseline: 1000 iters for σ → 0 reproduction of 28 cy
//
// Output: min / avg / max / σ over N runs for each config.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256

template<int CL>
__global__ __launch_bounds__(32, 1)
void local_lat(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;
    }
    __syncthreads();
    if (tid != 0) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(cur) : "r"(local_base + cur) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (blockIdx.x == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

template<int CX, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void dsmem_lat(unsigned long long* out, unsigned seed) {
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

    if (tid != 0) return;
    unsigned cur = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur) : "r"(peer_base + cur) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (my_cta == 0 && blockIdx.x == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

static void stats(const double* samples, int n, double& mn, double& avg, double& mx, double& sd) {
    mn = 1e30; mx = 0; avg = 0;
    for (int i = 0; i < n; i++) { if (samples[i]<mn) mn=samples[i]; if (samples[i]>mx) mx=samples[i]; avg+=samples[i]; }
    avg /= n;
    sd = 0;
    for (int i = 0; i < n; i++) sd += (samples[i]-avg)*(samples[i]-avg);
    sd = sqrt(sd/n);
}

template<int CL>
static int run_local(unsigned long long* d_out, double* samples_out, int N) {
    // warmup
    local_lat<CL><<<1, 32>>>(d_out, 42);
    CK(cudaDeviceSynchronize());
    for (int r = 0; r < N; r++) {
        local_lat<CL><<<1, 32>>>(d_out, 42 + r);
        CK(cudaDeviceSynchronize());
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        samples_out[r] = (double)cy / CL;
    }
    return 0;
}

template<int CX, int CL>
static int run_dsmem(unsigned long long* d_out, double* samples_out, int N) {
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

    int crashes = 0;
    int got = 0;
    for (int r = 0; r < N * 2 && got < N; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, dsmem_lat<CX, CL>, d_out, 42u + r);
        if (e) { crashes++; continue; }
        e = cudaDeviceSynchronize();
        if (e) { crashes++; continue; }
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        samples_out[got++] = (double)cy / CL;
    }
    if (crashes > 0) fprintf(stderr, "  (cx=%d cl=%d had %d crashes, got %d/%d samples)\n", CX, CL, crashes, got, N);
    return (got == N) ? 0 : -1;
}

int main() {
    CK(cudaSetDevice(0));
    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 16));
    double s[64];
    double mn, avg, mx, sd;

    printf("=== V12 DSMEM canonical latency (per-cluster CHAIN_LEN, 30 samples) ===\n");
    printf("(dependent pointer chain → DCE-immune; clock64-timed)\n\n");

    // --- LOCAL SMEM ---
    if (run_local<1000>(d_out, s, 30) == 0) {
        stats(s, 30, mn, avg, mx, sd);
        printf("Local SMEM (CL=1000, N=30):  min=%.2f avg=%.2f max=%.2f σ=%.3f cy/load\n", mn, avg, mx, sd);
    }

    // --- DSMEM cx=2 (CHAIN_LEN=50 safe) ---
    if (run_dsmem<2, 50>(d_out, s, 30) == 0) {
        stats(s, 30, mn, avg, mx, sd);
        printf("DSMEM cx=2 (CL=50, N=30):    min=%.2f avg=%.2f max=%.2f σ=%.3f cy/load\n", mn, avg, mx, sd);
    }

    // --- DSMEM cx=4 (CHAIN_LEN=5 safe) ---
    if (run_dsmem<4, 5>(d_out, s, 30) == 0) {
        stats(s, 30, mn, avg, mx, sd);
        printf("DSMEM cx=4 (CL=5, N=30):     min=%.2f avg=%.2f max=%.2f σ=%.3f cy/load\n", mn, avg, mx, sd);
    }

    // --- DSMEM cx=8 (CHAIN_LEN=5 safe) ---
    if (run_dsmem<8, 5>(d_out, s, 30) == 0) {
        stats(s, 30, mn, avg, mx, sd);
        printf("DSMEM cx=8 (CL=5, N=30):     min=%.2f avg=%.2f max=%.2f σ=%.3f cy/load\n", mn, avg, mx, sd);
    }

    // --- DSMEM cx=3,5,6,7 — fine cluster size sweep ---
    printf("\n--- Fine cluster sweep (CL=5) ---\n");
    if (run_dsmem<3, 5>(d_out, s, 20) == 0) {
        stats(s, 20, mn, avg, mx, sd);
        printf("DSMEM cx=3 (CL=5, N=20):     min=%.2f avg=%.2f max=%.2f σ=%.3f cy/load\n", mn, avg, mx, sd);
    }
    if (run_dsmem<5, 5>(d_out, s, 20) == 0) {
        stats(s, 20, mn, avg, mx, sd);
        printf("DSMEM cx=5 (CL=5, N=20):     min=%.2f avg=%.2f max=%.2f σ=%.3f cy/load\n", mn, avg, mx, sd);
    }
    if (run_dsmem<6, 5>(d_out, s, 20) == 0) {
        stats(s, 20, mn, avg, mx, sd);
        printf("DSMEM cx=6 (CL=5, N=20):     min=%.2f avg=%.2f max=%.2f σ=%.3f cy/load\n", mn, avg, mx, sd);
    }
    if (run_dsmem<7, 5>(d_out, s, 20) == 0) {
        stats(s, 20, mn, avg, mx, sd);
        printf("DSMEM cx=7 (CL=5, N=20):     min=%.2f avg=%.2f max=%.2f σ=%.3f cy/load\n", mn, avg, mx, sd);
    }

    cudaFree(d_out);
    return 0;
}
