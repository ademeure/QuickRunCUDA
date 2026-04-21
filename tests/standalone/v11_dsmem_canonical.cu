// V11: DSMEM canonical latency measurement — ONE kernel, all cluster sizes
// Resolves contradictions between:
//   - 04_dsmem_overhead.md (cluster=2: 224 cy, cluster=4/8: 201 cy, local: 28 cy)
//   - V5 M7 (DSMEM 214, local 54 — "54 cy local" was wrong)
//   - V10 attempts (rejected — DCE in static-offset BW tests)
//
// Methodology (DCE-immune per 04_dsmem):
//   - Dependent pointer chain: cur = load(peer_base + cur)
//   - smem init with byte offsets that stay bounded
//   - clock64 for timing (DCE-immune by construction)
//   - Anti-DCE output write of cur
//   - Crash mitigation: small CHAIN_LEN, multiple small runs
//
// Expected SASS:
//   Local: LDS R, [R+UR]
//   DSMEM: LD.E R, [R] (via global window — 04_dsmem finding)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

// Bound smem to 256 words (1 KB) — all offsets < 4 × 256 = 1024 bytes
#define SMEM_W 256
#define CHAIN_LEN 50    // 50 iters is safe for cluster=2 per 04_dsmem

// Local SMEM baseline
__global__ __launch_bounds__(32, 1)
void local_lat(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;

    // Init: each slot contains byte offset of next slot (mod SMEM_W)
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;  // byte offset
    }
    __syncthreads();

    if (tid != 0) return;

    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned cur = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < CHAIN_LEN; i++) {
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(cur) : "r"(local_base + cur) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;  // anti-DCE
    }
}

// DSMEM kernel — template on cluster size
template<int CX>
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
    for (int i = 0; i < CHAIN_LEN; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur) : "r"(peer_base + cur) : "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (my_cta == 0 && blockIdx.x == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = cur;
    }
}

static int run_local(unsigned long long* d_out, unsigned long long* result_cy) {
    local_lat<<<1, 32>>>(d_out, 42);
    CK(cudaDeviceSynchronize());
    local_lat<<<1, 32>>>(d_out, 42);
    CK(cudaDeviceSynchronize());
    cudaMemcpy(result_cy, d_out, 8, cudaMemcpyDeviceToHost);
    return 0;
}

template<int CX>
static int run_dsmem(unsigned long long* d_out, unsigned long long* result_cy) {
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

    CK(cudaLaunchKernelEx(&cfg, dsmem_lat<CX>, d_out, 42u));
    CK(cudaDeviceSynchronize());
    CK(cudaLaunchKernelEx(&cfg, dsmem_lat<CX>, d_out, 42u));
    CK(cudaDeviceSynchronize());
    cudaMemcpy(result_cy, d_out, 8, cudaMemcpyDeviceToHost);
    return 0;
}

int main() {
    CK(cudaSetDevice(0));
    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 16));

    unsigned long long cy;

    printf("=== V11 DSMEM canonical latency (CHAIN_LEN=%d, dep chain, DCE-immune) ===\n", CHAIN_LEN);

    if (run_local(d_out, &cy) == 0) {
        printf("  Local SMEM:     %.2f cy/load (%llu total)\n",
               (double)cy / CHAIN_LEN, cy);
    }

    // Cluster variants — attempt each, allow crash
    if (run_dsmem<2>(d_out, &cy) == 0) {
        printf("  DSMEM cx=2:     %.2f cy/load\n", (double)cy / CHAIN_LEN);
    }
    if (run_dsmem<4>(d_out, &cy) == 0) {
        printf("  DSMEM cx=4:     %.2f cy/load\n", (double)cy / CHAIN_LEN);
    }
    if (run_dsmem<8>(d_out, &cy) == 0) {
        printf("  DSMEM cx=8:     %.2f cy/load\n", (double)cy / CHAIN_LEN);
    }

    cudaFree(d_out);
    return 0;
}
