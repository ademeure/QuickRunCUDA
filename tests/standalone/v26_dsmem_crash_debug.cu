// V26: DSMEM varying-address crash debug
// 04_dsmem noted non-deterministic crashes with dependent chains beyond certain iter counts.
// Hypothesis: may be related to address computation overflow, or specific HW counter saturation.
//
// Systematically test at cluster=4 with increasing CL to find exact crash threshold.
// Also: try alternative address-computation patterns.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 4  // test with cluster=4 where 04_dsmem says crashes at 10+

// A) Baseline: dep chain as in V12
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void dep_chain(unsigned long long* out, unsigned seed) {
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
                 : "=r"(peer_base) : "r"(local_base), "r"((my_cta + 1u) % CX));

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

// B) Fixed-address (no chain) — should NOT crash
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void fixed_addr(unsigned long long* out, unsigned seed) {
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
                 : "=r"(peer_base) : "r"(local_base), "r"((my_cta + 1u) % CX));

    unsigned addr = peer_base + ((seed & 0xFF) << 2);
    unsigned long long t0 = 0, t1 = 0;
    unsigned cur = 0;

    if (my_cta == 0 && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(addr) : "memory");
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

// C) Strided addr (no chain, but varies each iter)
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void strided_addr(unsigned long long* out, unsigned seed) {
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
                 : "=r"(peer_base) : "r"(local_base), "r"((my_cta + 1u) % CX));

    unsigned long long t0 = 0, t1 = 0;
    unsigned cur = 0;

    if (my_cta == 0 && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            unsigned off = ((i * 37u + seed) & 0xFF) << 2;
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(cur) : "r"(peer_base + off) : "memory");
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

// D) Chain with SAT (saturating arithmetic) — might avoid overflow in address
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void sat_chain(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    for (int i = tid; i < SMEM_W; i += 32) {
        unsigned val = ((i + 1) * 37u + seed);
        smem[i] = (val & (SMEM_W - 1)) * 4u;  // bounded offset
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

    unsigned long long t0 = 0, t1 = 0;
    unsigned cur = 0;

    if (my_cta == 0 && tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            unsigned loaded;
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(loaded) : "r"(peer_base + cur) : "memory");
            // AND mask to guarantee bounded — same effect as bounded initial values
            cur = loaded & 0x3FC;
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

template<typename K>
static int run_many(K kernel, int N, int* crash_out, double* cy_out) {
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
    double sum = 0; int got = 0, crashes = 0;
    for (int r = 0; r < N; r++) {
        cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, d_out, 42u + r);
        if (e) { crashes++; cudaGetLastError(); continue; }
        e = cudaDeviceSynchronize();
        if (e) { crashes++; cudaGetLastError(); continue; }
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        sum += (double)cy;
        got++;
    }
    cudaFree(d_out);
    *crash_out = crashes;
    *cy_out = got > 0 ? sum / got : 0;
    return got;
}

int main() {
    CK(cudaSetDevice(0));
    double cy;
    int crashes;

    printf("=== V26 DSMEM crash debug (CX=4) — finding crash threshold by CL ===\n");
    printf("Each config: 30 attempts. Reports crash count + avg cy when successful.\n\n");

    printf("A) Dep-chain (matches V12/04_dsmem):\n");
    for (int cl : {5, 8, 10, 12, 15, 20, 30, 50}) {
        int n = 0;
        if (cl == 5) n = run_many(dep_chain<5>, 30, &crashes, &cy);
        if (cl == 8) n = run_many(dep_chain<8>, 30, &crashes, &cy);
        if (cl == 10) n = run_many(dep_chain<10>, 30, &crashes, &cy);
        if (cl == 12) n = run_many(dep_chain<12>, 30, &crashes, &cy);
        if (cl == 15) n = run_many(dep_chain<15>, 30, &crashes, &cy);
        if (cl == 20) n = run_many(dep_chain<20>, 30, &crashes, &cy);
        if (cl == 30) n = run_many(dep_chain<30>, 30, &crashes, &cy);
        if (cl == 50) n = run_many(dep_chain<50>, 30, &crashes, &cy);
        printf("  CL=%-3d: %2d/30 succ, %2d crashes, %5.0f cy avg (%.1f cy/load)\n",
               cl, n, crashes, cy, cy/cl);
    }

    printf("\nB) Fixed-address (no chain) — should never crash:\n");
    for (int cl : {10, 50, 100, 200, 500, 1000}) {
        int n = 0;
        if (cl == 10) n = run_many(fixed_addr<10>, 20, &crashes, &cy);
        if (cl == 50) n = run_many(fixed_addr<50>, 20, &crashes, &cy);
        if (cl == 100) n = run_many(fixed_addr<100>, 20, &crashes, &cy);
        if (cl == 200) n = run_many(fixed_addr<200>, 20, &crashes, &cy);
        if (cl == 500) n = run_many(fixed_addr<500>, 20, &crashes, &cy);
        if (cl == 1000) n = run_many(fixed_addr<1000>, 20, &crashes, &cy);
        printf("  CL=%-4d: %2d/20 succ, %2d crashes, %5.0f cy avg\n", cl, n, crashes, cy);
    }

    printf("\nC) Strided-address (addr varies each iter, no chain):\n");
    for (int cl : {10, 50, 100, 200, 500}) {
        int n = 0;
        if (cl == 10) n = run_many(strided_addr<10>, 20, &crashes, &cy);
        if (cl == 50) n = run_many(strided_addr<50>, 20, &crashes, &cy);
        if (cl == 100) n = run_many(strided_addr<100>, 20, &crashes, &cy);
        if (cl == 200) n = run_many(strided_addr<200>, 20, &crashes, &cy);
        if (cl == 500) n = run_many(strided_addr<500>, 20, &crashes, &cy);
        printf("  CL=%-4d: %2d/20 succ, %2d crashes, %5.0f cy avg\n", cl, n, crashes, cy);
    }

    printf("\nD) SAT-chain (chain with AND-masked cur):\n");
    for (int cl : {5, 10, 20, 50, 100, 200}) {
        int n = 0;
        if (cl == 5) n = run_many(sat_chain<5>, 20, &crashes, &cy);
        if (cl == 10) n = run_many(sat_chain<10>, 20, &crashes, &cy);
        if (cl == 20) n = run_many(sat_chain<20>, 20, &crashes, &cy);
        if (cl == 50) n = run_many(sat_chain<50>, 20, &crashes, &cy);
        if (cl == 100) n = run_many(sat_chain<100>, 20, &crashes, &cy);
        if (cl == 200) n = run_many(sat_chain<200>, 20, &crashes, &cy);
        printf("  CL=%-4d: %2d/20 succ, %2d crashes, %5.0f cy avg\n", cl, n, crashes, cy);
    }

    return 0;
}
