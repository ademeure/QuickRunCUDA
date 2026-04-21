// V29: TMA tile-size sweep + ILP up to 16 + atomic CAS
// A) TMA multicast at tile sizes 1K, 4K, 16K, 64K
// B) ILP={12, 16, 32} to find per-CTA read BW ceiling
// C) atom.cas.shared::cluster vs atom.add
// D) DSMEM reads while peer is doing heavy compute (does it starve?)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define CX 8

// A) TMA multicast for various tile sizes
template<int TILE_BYTES>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(128, 1)
void tma_tile_size(unsigned long long* out, const float* src) {
    extern __shared__ __align__(16) char smem[];
    __shared__ __align__(8) unsigned long long mbar;

    int tid = threadIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                     :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (my_cta == 0 && tid == 0) {
        const uint16_t mask = (uint16_t)((1u << CX) - 1);
        unsigned buf_addr = (unsigned)__cvta_generic_to_shared(&smem[0]);
        unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                     :: "r"(mbar_addr), "r"(TILE_BYTES) : "memory");
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1], %2, [%3], %4;\n"
            :: "r"(buf_addr), "l"(src), "r"(TILE_BYTES), "r"(mbar_addr), "h"(mask)
            : "memory");
    }

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

    if (tid == 0) ((float*)&out[2 + my_cta])[0] = ((float*)smem)[0];

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

// B) Ring read with ILP up to 32
template<int ILP, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void ring_rd_ilp(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[1024];
    int tid = threadIdx.x;
    for (int i = tid; i < 1024; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (1024 - 1)) * 4u;
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

    unsigned c[32];
    #pragma unroll
    for (int k = 0; k < ILP; k++) c[k] = (tid * 4u + k * 32u) & (1024*4 - 1);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(c[k]) : "r"(peer_base + c[k]) : "memory");
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

// C) atomic CAS
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void atom_cas_test(unsigned long long* out, unsigned seed) {
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
                 : "=r"(peer_base) : "r"(local_base), "r"(1u));

    unsigned addr = peer_base + (seed & 0xFC);
    unsigned acc = seed;
    unsigned long long t0 = 0, t1 = 0;

    if (my_cta == 0 && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("atom.cas.shared::cluster.b32 %0, [%1], %2, %3;"
                         : "=r"(acc) : "r"(addr), "r"(acc), "r"(acc + 1) : "memory");
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

// D) Peer under compute load — does DSMEM get starved?
template<int CL, int COMPUTE_ITERS>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void peer_under_compute(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[256];
    int tid = threadIdx.x;
    for (int i = tid; i < 256; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (256 - 1)) * 4u;
    }
    __syncthreads();

    unsigned my_cta;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(1u));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (my_cta == 0) {
        // CTA 0 does DSMEM reads
        if (tid == 0) {
            unsigned cur = 0;
            #pragma unroll 1
            for (int i = 0; i < CL; i++) {
                asm volatile("ld.shared::cluster.u32 %0, [%1];"
                             : "=r"(cur) : "r"(peer_base + cur) : "memory");
            }
            out[2] = cur;
        }
    } else if (my_cta == 1) {
        // CTA 1 does pure FFMA to burn compute
        float a = 1.1f, b = 2.2f, c = 3.3f;
        #pragma unroll 1
        for (int i = 0; i < COMPUTE_ITERS; i++) {
            asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                         : "=f"(c) : "f"(a), "f"(b), "f"(c));
            asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                         : "=f"(a) : "f"(c), "f"(b), "f"(a));
            asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                         : "=f"(b) : "f"(a), "f"(c), "f"(b));
        }
        if (tid == 0) ((float*)&out[3])[0] = a + b + c;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) out[0] = t1 - t0;
}

template<typename K, typename... Args>
static int avg_cy_t(int threads, int shmem_bytes, K kernel, int N, double* cy_out, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CX, 1, 1);
    cfg.blockDim = dim3(threads, 1, 1);
    cfg.stream = 0;
    cfg.dynamicSmemBytes = shmem_bytes;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CX;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    unsigned long long* d_out;
    cudaMalloc(&d_out, 256);
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

    // Allocate source for TMA
    unsigned N = 65536 * CX;  // up to 64K bytes × 8
    float* d_src;
    CK(cudaMalloc(&d_src, N * 4));
    cudaMemset(d_src, 0, N * 4);

    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    double cy;

    printf("=== V29 DSMEM extreme tests (CX=8, 1920 MHz) ===\n\n");

    // A) TMA tile sweep
    printf("A) TMA multicast tile size sweep (8-way):\n");
    #define TMA(SZ) { \
        int got = avg_cy_t(128, SZ, tma_tile_size<SZ>, 15, &cy, (const float*)d_src); \
        if (got > 0) { \
            double t_us = cy / CLOCK_GHZ / 1e3; \
            double bytes_per_cta = SZ; \
            double delivered = bytes_per_cta * CX; \
            double bw_gbps = delivered / (cy / CLOCK_GHZ / 1e9) / 1e9; \
            printf("  tile=%-5d B: %.0f cy (%.2f us), effective BW=%.2f GB/s\n", \
                   SZ, cy, t_us, bw_gbps); \
        } }
    TMA(1024); TMA(4096); TMA(16384); TMA(32768); TMA(65536);
    #undef TMA

    // B) ILP sweep
    printf("\nB) Ring read ILP sweep (CL=5, 1 warp per CTA):\n");
    int got;
    got = avg_cy_t(32, 0, ring_rd_ilp<1, 5>, 15, &cy, 42u);
    if (got > 0) { double ops = 32*1*5; double t = cy/CLOCK_GHZ/1e9; printf("  ILP=1:  %.0f cy (%.2f cy/ld), %.2f GB/s per CTA\n", cy, cy/ops, ops*4/t/1e9); }
    got = avg_cy_t(32, 0, ring_rd_ilp<4, 5>, 15, &cy, 42u);
    if (got > 0) { double ops = 32*4*5; double t = cy/CLOCK_GHZ/1e9; printf("  ILP=4:  %.0f cy (%.2f cy/ld), %.2f GB/s per CTA\n", cy, cy/ops, ops*4/t/1e9); }
    got = avg_cy_t(32, 0, ring_rd_ilp<8, 5>, 15, &cy, 42u);
    if (got > 0) { double ops = 32*8*5; double t = cy/CLOCK_GHZ/1e9; printf("  ILP=8:  %.0f cy (%.2f cy/ld), %.2f GB/s per CTA\n", cy, cy/ops, ops*4/t/1e9); }
    got = avg_cy_t(32, 0, ring_rd_ilp<12, 5>, 15, &cy, 42u);
    if (got > 0) { double ops = 32*12*5; double t = cy/CLOCK_GHZ/1e9; printf("  ILP=12: %.0f cy (%.2f cy/ld), %.2f GB/s per CTA\n", cy, cy/ops, ops*4/t/1e9); }
    got = avg_cy_t(32, 0, ring_rd_ilp<16, 5>, 15, &cy, 42u);
    if (got > 0) { double ops = 32*16*5; double t = cy/CLOCK_GHZ/1e9; printf("  ILP=16: %.0f cy (%.2f cy/ld), %.2f GB/s per CTA\n", cy, cy/ops, ops*4/t/1e9); }

    // C) atom.cas
    printf("\nC) Atomic CAS vs add (CL=20, 1 thread):\n");
    got = avg_cy_t(32, 0, atom_cas_test<20>, 20, &cy, 42u);
    if (got > 0) printf("  atom.cas.shared::cluster.b32: %.0f cy (%.2f cy/CAS)\n", cy, cy/20);

    // D) Peer compute load
    printf("\nD) DSMEM read while peer busy with FFMA (10000 iters):\n");
    got = avg_cy_t(32, 0, peer_under_compute<50, 10000>, 15, &cy, 42u);
    if (got > 0) printf("  CL=50 reads, peer FFMA: %.0f cy (%.2f cy/load)\n", cy, cy/50);
    got = avg_cy_t(32, 0, peer_under_compute<50, 0>, 15, &cy, 42u);
    if (got > 0) printf("  CL=50 reads, peer idle: %.0f cy (%.2f cy/load)\n", cy, cy/50);

    cudaFree(d_src);
    cudaFree(d_out);
    return 0;
}
