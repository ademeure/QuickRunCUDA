// V48: TMA multicast PIPELINED N-deep across 18 clusters
// V32 single TMA per iter: 14.9 TB/s effective
// V46 read pipelined 8: 7.20 TB/s raw HBM (98%)
// Hypothesis: multicast + 8-deep = 8 × 7.20 / 1 = 57.6 TB/s effective?
// Reality limit: 14.9 was multicast-bound by TMA issue rate; pipelining might 8×

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define CX 8

template<int TILE_BYTES, int N_INFLIGHT, int N_ITERS>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(128, 1)
void mcast_pipelined(const float* src, unsigned long long* out) {
    extern __shared__ __align__(16) char buf_raw[];
    __shared__ __align__(8) unsigned long long mbar[16];

    int tid = threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < N_INFLIGHT; i++) {
            asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                         :: "r"((unsigned)__cvta_generic_to_shared(&mbar[i])) : "memory");
        }
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned bufs[16];
    unsigned mbars[16];
    #pragma unroll
    for (int i = 0; i < N_INFLIGHT; i++) {
        bufs[i] = (unsigned)__cvta_generic_to_shared(&buf_raw[i * TILE_BYTES]);
        mbars[i] = (unsigned)__cvta_generic_to_shared(&mbar[i]);
    }
    const uint16_t mask = (uint16_t)((1u << CX) - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        // ALL CTAs arm their local mbarriers
        if (tid == 0) {
            #pragma unroll
            for (int i = 0; i < N_INFLIGHT; i++) {
                asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbars[i]) : "memory");
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                             :: "r"(mbars[i]), "r"(TILE_BYTES) : "memory");
            }
        }
        __syncthreads();
        // CTA 0 issues all N_INFLIGHT multicast TMAs
        if (my_cta == 0 && tid == 0) {
            #pragma unroll
            for (int i = 0; i < N_INFLIGHT; i++) {
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                    " [%0], [%1], %2, [%3], %4;\n"
                    :: "r"(bufs[i]), "l"(src + i * TILE_BYTES / 4), "r"(TILE_BYTES), "r"(mbars[i]), "h"(mask)
                    : "memory");
            }
        }
        // All CTAs wait for all N_INFLIGHT multicasts
        if (tid == 0) {
            #pragma unroll
            for (int i = 0; i < N_INFLIGHT; i++) {
                int done = 0; int spin = 0;
                while (!done && spin < 1000000) {
                    asm volatile("{.reg .pred p;\n"
                                 "mbarrier.try_wait.shared.b64 p, [%1], 0;\n"
                                 "selp.u32 %0, 1, 0, p;}\n"
                                 : "=r"(done) : "r"(mbars[i]) : "memory");
                    spin++;
                }
            }
        }
        __syncthreads();
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0 && blockIdx.x == 0) {
        out[0] = t1 - t0;
        ((float*)&out[2])[0] = ((float*)buf_raw)[0];
    }
}

int main() {
    CK(cudaSetDevice(0));

    size_t words = 1ull * 1024 * 1024 * 1024;
    float* d_src;
    CK(cudaMalloc(&d_src, words * 4));
    cudaMemset(d_src, 0, words * 4);

    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 256;
    const int TILE = 32768;
    const int N_CLUSTERS = 18;

    printf("=== V48 TMA multicast pipelined N-deep (18 clusters × 8 CTAs × %d KB tile × %d iters) ===\n", TILE/1024, N_ITERS);
    printf("N_inflight  shmem_KB  wall_ms   raw_HBM_TB/s   effective_TB/s   per_iter_us\n");

    auto run = [&](int n_inflight) {
        int shmem = n_inflight * TILE;
        if (shmem > 163840) { printf("%2d         too big\n", n_inflight); return; }

        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(N_CLUSTERS * CX, 1, 1);
        cfg.blockDim = dim3(128, 1, 1);
        cfg.dynamicSmemBytes = shmem;
        cfg.stream = 0;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = CX;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;

        cudaError_t e;

        if (n_inflight == 1) {
            cudaFuncSetAttribute(mcast_pipelined<32768, 1, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
            e = cudaLaunchKernelEx(&cfg, mcast_pipelined<32768, 1, 256>, (const float*)d_src, d_out);
        } else if (n_inflight == 2) {
            cudaFuncSetAttribute(mcast_pipelined<32768, 2, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
            e = cudaLaunchKernelEx(&cfg, mcast_pipelined<32768, 2, 256>, (const float*)d_src, d_out);
        } else if (n_inflight == 4) {
            cudaFuncSetAttribute(mcast_pipelined<32768, 4, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
            e = cudaLaunchKernelEx(&cfg, mcast_pipelined<32768, 4, 256>, (const float*)d_src, d_out);
        } else if (n_inflight == 8) {
            cudaFuncSetAttribute(mcast_pipelined<TILE, 8, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
            e = cudaLaunchKernelEx(&cfg, mcast_pipelined<TILE, 8, 256>, (const float*)d_src, d_out);
        } else { return; }
        if (e) { printf("%2d LAUNCH-FAIL\n", n_inflight); cudaGetLastError(); return; }
        e = cudaDeviceSynchronize();
        if (e) { printf("%2d EXEC-FAIL\n", n_inflight); cudaGetLastError(); return; }

        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (n_inflight == 1) cudaLaunchKernelEx(&cfg, mcast_pipelined<32768, 1, 256>, (const float*)d_src, d_out);
            else if (n_inflight == 2) cudaLaunchKernelEx(&cfg, mcast_pipelined<32768, 2, 256>, (const float*)d_src, d_out);
            else if (n_inflight == 4) cudaLaunchKernelEx(&cfg, mcast_pipelined<32768, 4, 256>, (const float*)d_src, d_out);
            else if (n_inflight == 8) cudaLaunchKernelEx(&cfg, mcast_pipelined<TILE, 8, 256>, (const float*)d_src, d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double hbm_bytes = (double)N_CLUSTERS * N_ITERS * n_inflight * TILE;
        double delivered = hbm_bytes * CX;
        double hbm_tbs = hbm_bytes / (avg_ms / 1e3) / 1e12;
        double eff_tbs = delivered / (avg_ms / 1e3) / 1e12;
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        double per_iter_us = cy / (double)N_ITERS / 1.920 / 1e3;
        printf("%2d         %5d     %7.3f   %8.3f       %8.3f         %5.2f\n",
               n_inflight, shmem/1024, avg_ms, hbm_tbs, eff_tbs, per_iter_us);
    };

    for (int n : {1, 2, 4, 8}) run(n);

    cudaFree(d_src);
    cudaFree(d_out);
    return 0;
}
