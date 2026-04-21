// V18: DSMEM under high load — does contention emerge at high ILP/threads?
//
// Tests N=1..8 active CTAs at:
//   - 32 threads × ILP=1
//   - 32 threads × ILP=4
//   - 32 threads × ILP=8
// Each thread does dep chain. Per-CTA wall cycles (not per-load) is the metric.
//
// Question: BW per CTA scales linearly with N or saturates?

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 8
#define CL 5

template<int N_ACTIVE, int ILP>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void load_test(unsigned long long* out, unsigned seed) {
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

    unsigned c[8];
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        c[k] = (tid * 4u + k * 32u) & (SMEM_W*4 - 1);
    }

    bool active = (my_cta < N_ACTIVE);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (active) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                asm volatile("ld.shared::cluster.u32 %0, [%1];"
                             : "=r"(c[k]) : "r"(peer_base + c[k]) : "memory");
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= c[k];
        ((unsigned*)out)[2] = acc;
    }
}

template<typename K>
static int avg_cy(K kernel, int N, double* cy_out) {
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
        cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, d_out, 42u + r);
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

    printf("=== V18 DSMEM load test (CL=5, 1920 MHz) ===\n");
    printf("Effective threads/CTA = 32, ILP per thread varied. Reports per-CTA cy + agg BW.\n\n");

    for (int ilp_idx = 0; ilp_idx < 3; ilp_idx++) {
        int ilp = (ilp_idx == 0) ? 1 : (ilp_idx == 1 ? 4 : 8);
        printf("--- ILP=%d ---\n", ilp);
        printf("  N    cy_per_CTA  cy/load    BW_total(GB/s)  BW_per_CTA(GB/s)  Slowdown\n");

        double cy_n1 = 0;
        for (int n = 1; n <= CX; n++) {
            int got = 0;
            if (ilp == 1) {
                if (n == 1) got = avg_cy(load_test<1,1>, 20, &cy);
                if (n == 2) got = avg_cy(load_test<2,1>, 20, &cy);
                if (n == 3) got = avg_cy(load_test<3,1>, 20, &cy);
                if (n == 4) got = avg_cy(load_test<4,1>, 20, &cy);
                if (n == 5) got = avg_cy(load_test<5,1>, 20, &cy);
                if (n == 6) got = avg_cy(load_test<6,1>, 20, &cy);
                if (n == 7) got = avg_cy(load_test<7,1>, 20, &cy);
                if (n == 8) got = avg_cy(load_test<8,1>, 20, &cy);
            } else if (ilp == 4) {
                if (n == 1) got = avg_cy(load_test<1,4>, 20, &cy);
                if (n == 2) got = avg_cy(load_test<2,4>, 20, &cy);
                if (n == 3) got = avg_cy(load_test<3,4>, 20, &cy);
                if (n == 4) got = avg_cy(load_test<4,4>, 20, &cy);
                if (n == 5) got = avg_cy(load_test<5,4>, 20, &cy);
                if (n == 6) got = avg_cy(load_test<6,4>, 20, &cy);
                if (n == 7) got = avg_cy(load_test<7,4>, 20, &cy);
                if (n == 8) got = avg_cy(load_test<8,4>, 20, &cy);
            } else {
                if (n == 1) got = avg_cy(load_test<1,8>, 20, &cy);
                if (n == 2) got = avg_cy(load_test<2,8>, 20, &cy);
                if (n == 3) got = avg_cy(load_test<3,8>, 20, &cy);
                if (n == 4) got = avg_cy(load_test<4,8>, 20, &cy);
                if (n == 5) got = avg_cy(load_test<5,8>, 20, &cy);
                if (n == 6) got = avg_cy(load_test<6,8>, 20, &cy);
                if (n == 7) got = avg_cy(load_test<7,8>, 20, &cy);
                if (n == 8) got = avg_cy(load_test<8,8>, 20, &cy);
            }
            if (got > 0) {
                double loads_per_cta = 32.0 * ilp * CL;
                double total_loads = loads_per_cta * n;
                double bytes_total = total_loads * 4.0;
                double time_s = cy / CLOCK_GHZ / 1e9;
                double bw_total = bytes_total / time_s / 1e9;
                double bw_per = bw_total / n;
                if (n == 1) cy_n1 = cy;
                printf("  %d    %8.0f   %6.2f    %10.2f       %8.2f         %.2fx\n",
                       n, cy, cy/loads_per_cta, bw_total, bw_per, cy/cy_n1);
            }
        }
        printf("\n");
    }

    return 0;
}
