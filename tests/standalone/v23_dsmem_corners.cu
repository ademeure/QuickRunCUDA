// V23: DSMEM corner cases
// A) Self-read via mapa (CTA 0 reads its own SMEM via mapa(me))
// B) Stride patterns / bank conflicts
// C) DSMEM + local SMEM concurrent (does DSMEM steal local SMEM BW?)
// D) Atomic scope comparison (local SMEM: .cta vs .cluster)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

#define SMEM_W 256
#define CX 8

// A) Self-read via mapa(me)
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void self_read_mapa(unsigned long long* out, unsigned seed) {
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
    unsigned self_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(self_base) : "r"(local_base), "r"(my_cta));

    if (my_cta != 0 || tid != 0) {
        asm volatile("barrier.cluster.arrive;" ::: "memory");
        asm volatile("barrier.cluster.wait;"  ::: "memory");
        return;
    }

    unsigned cur = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(cur) : "r"(self_base + cur) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    out[0] = t1 - t0;
    ((unsigned*)out)[2] = cur;
}

// B) Bank conflict test: different strides
template<int STRIDE, int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void strided_rd(unsigned long long* out, unsigned seed) {
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

    // Each thread accesses address = tid * STRIDE (mod SMEM)
    unsigned addr = peer_base + ((tid * STRIDE) & (SMEM_W*4 - 4));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    unsigned v = 0;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("ld.shared::cluster.u32 %0, [%1];"
                     : "=r"(v) : "r"(addr + (i & 0xFC)) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = v;
    }
}

// C) DSMEM + local SMEM concurrent read
template<int CL>
__global__ __cluster_dims__(CX, 1, 1) __launch_bounds__(32, 1)
void dsmem_plus_local(unsigned long long* out, unsigned seed) {
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

    // Half the threads read local SMEM, half DSMEM
    bool is_local = (tid < 16);

    unsigned c0 = (tid * 4u) & (SMEM_W*4 - 1);
    unsigned c1 = (tid * 4u + 32u) & (SMEM_W*4 - 1);

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (is_local) {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared.u32 %0, [%1];"
                         : "=r"(c0) : "r"(local_base + c0) : "memory");
            asm volatile("ld.shared.u32 %0, [%1];"
                         : "=r"(c1) : "r"(local_base + c1) : "memory");
        }
    } else {
        #pragma unroll 1
        for (int i = 0; i < CL; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(c0) : "r"(peer_base + c0) : "memory");
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(c1) : "r"(peer_base + c1) : "memory");
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    if (my_cta == 0 && tid == 0) {
        out[0] = t1 - t0;
        ((unsigned*)out)[2] = c0 ^ c1;
    }
    if (my_cta == 0 && tid == 16) {
        out[1] = t1 - t0;
    }
}

// D) Atom scope on LOCAL smem
template<int CL>
__global__ __launch_bounds__(32, 1)
void atom_local_cta(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    if (tid < SMEM_W) smem[tid] = 0;
    __syncthreads();

    if (tid != 0) return;

    unsigned addr = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("atom.add.shared.u32 %0, [%1], %2;"
                     : "=r"(acc) : "r"(addr), "r"(acc + i) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    out[0] = t1 - t0;
    ((unsigned*)out)[2] = acc;
}

template<int CL>
__global__ __launch_bounds__(32, 1)
void atom_local_cluster(unsigned long long* out, unsigned seed) {
    __shared__ unsigned int smem[SMEM_W];
    int tid = threadIdx.x;
    if (tid < SMEM_W) smem[tid] = 0;
    __syncthreads();

    if (tid != 0) return;

    unsigned addr = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < CL; i++) {
        asm volatile("atom.add.shared::cluster.u32 %0, [%1], %2;"
                     : "=r"(acc) : "r"(addr), "r"(acc + i) : "memory");
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    out[0] = t1 - t0;
    ((unsigned*)out)[2] = acc;
}

template<typename K, typename... Args>
static int avg_cy_cluster(K kernel, int N, double* cy_out, Args... args) {
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

template<typename K, typename... Args>
static int avg_cy_regular(K kernel, int N, double* cy_out, Args... args) {
    unsigned long long* d_out;
    cudaMalloc(&d_out, 16);
    double sum = 0; int got = 0;
    for (int r = 0; r < N; r++) {
        kernel<<<1, 32>>>(d_out, args..., 42u + r);
        cudaError_t e = cudaDeviceSynchronize();
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

    printf("=== V23 DSMEM corners ===\n\n");

    // A) Self-read via mapa
    printf("A) Self-read via mapa (CTA 0 mapa to itself):\n");
    int got = avg_cy_cluster(self_read_mapa<50>, 20, &cy);
    if (got > 0) printf("  CL=50: %.0f cy (%.2f cy/load) — compare to local SMEM 24 cy\n",
                        cy, cy/50);

    // B) Stride / bank conflicts
    printf("\nB) Stride patterns (ring rd, 32t, CL=50):\n");
    got = avg_cy_cluster(strided_rd<4, 50>, 15, &cy);
    if (got > 0) printf("  stride=4B   (no conflict): %.0f cy (%.2f cy/load)\n", cy, cy/50);
    got = avg_cy_cluster(strided_rd<8, 50>, 15, &cy);
    if (got > 0) printf("  stride=8B   (2-way):       %.0f cy (%.2f cy/load)\n", cy, cy/50);
    got = avg_cy_cluster(strided_rd<16, 50>, 15, &cy);
    if (got > 0) printf("  stride=16B  (4-way):       %.0f cy (%.2f cy/load)\n", cy, cy/50);
    got = avg_cy_cluster(strided_rd<32, 50>, 15, &cy);
    if (got > 0) printf("  stride=32B  (8-way):       %.0f cy (%.2f cy/load)\n", cy, cy/50);
    got = avg_cy_cluster(strided_rd<128, 50>, 15, &cy);
    if (got > 0) printf("  stride=128B (all-same/32-way): %.0f cy (%.2f cy/load)\n", cy, cy/50);

    // C) DSMEM + local concurrent
    printf("\nC) DSMEM + local SMEM concurrent (half warp each):\n");
    got = avg_cy_cluster(dsmem_plus_local<50>, 15, &cy);
    if (got > 0) printf("  CL=50: %.0f cy total, cy/load (local half) %.2f\n", cy, cy/100);

    // D) Atomic scope on local SMEM
    printf("\nD) Local SMEM atomic scope:\n");
    got = avg_cy_regular(atom_local_cta<100>, 20, &cy);
    if (got > 0) printf("  atom.add.shared.u32           (.cta):     %.0f cy (%.2f cy/atom)\n", cy, cy/100);
    got = avg_cy_regular(atom_local_cluster<100>, 20, &cy);
    if (got > 0) printf("  atom.add.shared::cluster.u32  (.cluster): %.0f cy (%.2f cy/atom)\n", cy, cy/100);

    return 0;
}
