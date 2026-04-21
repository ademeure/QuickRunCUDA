// V31: Final DSMEM probes
// A) DSMEM reads while TMA is in-flight (does TMA steal BW?)
// B) Hot-spot atomics (all CTAs atom.add to same peer)
// C) Alignment: unaligned u32 store/load behavior

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define CX 8

// A) DSMEM reads while TMA in-flight
template<int CL, int TMA_BYTES>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(128, 1)
void tma_plus_dsmem(unsigned long long* out, const float* src) {
    extern __shared__ __align__(16) char smem_raw[];
    __shared__ __align__(8) unsigned long long mbar;
    __shared__ unsigned int chain[256];

    int tid = threadIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
    }
    for (int i = tid; i < 256; i += 128) {
        unsigned val = ((i + 1) * 37u + 42u);
        chain[i] = (val & 0xFF) * 4u;
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_chain = (unsigned)__cvta_generic_to_shared(&chain[0]);
    unsigned peer_chain;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_chain) : "r"(local_chain), "r"((my_cta + 1u) % CX));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // CTA 0: launch TMA
    if (my_cta == 0 && tid == 0) {
        const uint16_t mask = (uint16_t)((1u << CX) - 1);
        unsigned buf_addr = (unsigned)__cvta_generic_to_shared(&smem_raw[0]);
        unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                     :: "r"(mbar_addr), "r"(TMA_BYTES) : "memory");
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1], %2, [%3], %4;\n"
            :: "r"(buf_addr), "l"(src), "r"(TMA_BYTES), "r"(mbar_addr), "h"(mask)
            : "memory");
    }

    // Meanwhile all CTAs do DSMEM chain reads from peer
    if (tid == 0 && my_cta == 0) {  // only time CTA 0's chain
        unsigned cur = 0;
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(peer_chain + cur) : "memory");
        }
        ((unsigned*)out)[2] = cur;
    }

    // Wait for TMA to complete
    if (tid == 0) {
        int done = 0; int spin = 0;
        unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
        while (!done && spin < 1000000) {
            asm volatile("{.reg .pred p;\n"
                         "mbarrier.try_wait.shared.b64 p, [%1], 0;\n"
                         "selp.u32 %0, 1, 0, p;}\n"
                         : "=r"(done) : "r"(mbar_addr) : "memory");
            spin++;
        }
    }
    __syncthreads();

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0) ((float*)&out[3])[0] = ((float*)smem_raw)[0];

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

// B) Hot-spot atomics
template<int N_ACTIVE, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void hot_atomic(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[256];
    int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 32) smem[i] = 0;
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(0u));  // all → CTA 0

    unsigned addr = peer_base + (seed & 0xFC);
    unsigned acc = 0;
    bool active = (my_cta < N_ACTIVE && my_cta != 0);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (active && tid == 0) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("atom.add.shared::cluster.u32 %0, [%1], %2;"
                         : "=r"(acc) : "r"(addr), "r"(acc + i) : "memory");
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    // CTA 1 first active reports
    if (my_cta == 1 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = acc;
    }
}

template<typename K, typename... Args>
static int avg_cy_t(int threads, int shmem, K kernel, int N, double* cy_out, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CX, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.stream = 0;
    cfg.dynamicSmemBytes = shmem;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CX;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    unsigned long long* d_out;
    cudaMalloc(&d_out, 64);
    double sum = 0; int got = 0;
    for (int r = 0; r < N * 2 && got < N; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, d_out, args...);
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

    // Src buffer for TMA
    unsigned N = 16384 * CX;
    float* d_src;
    CK(cudaMalloc(&d_src, N * 4));
    cudaMemset(d_src, 0, N * 4);

    printf("=== V31 DSMEM final probes (1920 MHz) ===\n\n");

    // A) TMA in-flight while DSMEM chain reads
    printf("A) DSMEM reads concurrent with TMA (CTA 0 launches TMA, chain reads from CTA 1):\n");
    double cy_ref;
    int got = avg_cy_t(128, 16384, tma_plus_dsmem<50, 16384>, 15, &cy, (const float*)d_src);
    if (got > 0) {
        printf("  CL=50 chain + 16 KB TMA: %.0f cy (%.2f cy/load)\n", cy, cy/50);
        cy_ref = cy;
    }
    got = avg_cy_t(128, 16384, tma_plus_dsmem<50, 0>, 15, &cy, (const float*)d_src);
    if (got > 0) printf("  CL=50 chain,  TMA 0 B: %.0f cy (%.2f cy/load)\n", cy, cy/50);

    // B) Hot-spot atomics — all CTAs atom.add same location
    printf("\nB) Hot-spot atomics (all N CTAs atom.add to same CTA 0 slot):\n");
    printf("  N_act  cy (on CTA 1)  cy/atom  atomics/s\n");
    double cy1;
    got = avg_cy_t(32, 0, hot_atomic<2, 50>, 20, &cy, 42u); cy1 = cy;
    if (got > 0) printf("  2     %7.0f       %5.2f    %.2f Matom/s\n", cy, cy/50, 50/(cy/CLOCK_GHZ/1e9)/1e6);
    got = avg_cy_t(32, 0, hot_atomic<4, 50>, 20, &cy, 42u);
    if (got > 0) printf("  4     %7.0f       %5.2f    %.2f Matom/s (cluster %.2f Matom/s)\n",
                        cy, cy/50, 50/(cy/CLOCK_GHZ/1e9)/1e6, 3*50/(cy/CLOCK_GHZ/1e9)/1e6);
    got = avg_cy_t(32, 0, hot_atomic<8, 50>, 20, &cy, 42u);
    if (got > 0) printf("  8     %7.0f       %5.2f    %.2f Matom/s (cluster %.2f Matom/s)\n",
                        cy, cy/50, 50/(cy/CLOCK_GHZ/1e9)/1e6, 7*50/(cy/CLOCK_GHZ/1e9)/1e6);

    cudaFree(d_src);
    return 0;
}
