// V8: Cluster DSMEM aggregate BW (standalone)
// Measures bandwidth when cluster CTAs read from each others' SMEM.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define CLUSTER_X 8
#define SMEM_WORDS 2048   // 8 KB
#define BOUND_MASK 4095
#define N_BLOCKS 144       // 18 clusters × 8 CTAs (near-full GPU)

__global__ __cluster_dims__(CLUSTER_X, 1, 1) __launch_bounds__(128, 1)
void kernel_dsmem(unsigned int* out, int ITERS, int seed) {
    __shared__ unsigned int smem[SMEM_WORDS];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    // Init SMEM with valid aligned offsets (for pointer-chase)
    #pragma unroll
    for (int i = tid; i < SMEM_WORDS; i += blockDim.x) {
        smem[i] = (((unsigned)i * 37) & (BOUND_MASK >> 2)) << 2;  // bytes 0, 4, ..., 4092
    }
    __syncthreads();

    unsigned my_cta, cluster_size;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_size));

    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    unsigned target_cta = (my_cta + 1u) % cluster_size;
    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    // Fixed per-thread offset; 8-way ILP inner unroll to saturate bandwidth.
    // Each load reads from compile-time-constant offset from peer_base + tid*4.
    unsigned base_u = peer_base + ((tid * 4) & BOUND_MASK);
    unsigned s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    unsigned s4 = 0, s5 = 0, s6 = 0, s7 = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("ld.shared::cluster.u32 %0, [%1+0];"   : "=r"(r0) : "r"(base_u));
        asm volatile("ld.shared::cluster.u32 %0, [%1+32];"  : "=r"(r1) : "r"(base_u));
        asm volatile("ld.shared::cluster.u32 %0, [%1+64];"  : "=r"(r2) : "r"(base_u));
        asm volatile("ld.shared::cluster.u32 %0, [%1+96];"  : "=r"(r3) : "r"(base_u));
        asm volatile("ld.shared::cluster.u32 %0, [%1+128];" : "=r"(r4) : "r"(base_u));
        asm volatile("ld.shared::cluster.u32 %0, [%1+160];" : "=r"(r5) : "r"(base_u));
        asm volatile("ld.shared::cluster.u32 %0, [%1+192];" : "=r"(r6) : "r"(base_u));
        asm volatile("ld.shared::cluster.u32 %0, [%1+224];" : "=r"(r7) : "r"(base_u));
        s0 += r0; s1 += r1; s2 += r2; s3 += r3;
        s4 += r4; s5 += r5; s6 += r6; s7 += r7;
    }
    s0 = s0^s1^s2^s3^s4^s5^s6^s7;

    unsigned sum = s0;
    out[gtid] = sum;
}

// Local-SMEM comparison kernel (same pattern, just ld.shared)
__global__ __launch_bounds__(128, 1)
void kernel_local(unsigned int* out, int ITERS, int seed) {
    __shared__ unsigned int smem[SMEM_WORDS];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    #pragma unroll
    for (int i = tid; i < SMEM_WORDS; i += blockDim.x) {
        smem[i] = ((unsigned)(gtid + i) * 2654435761u) ^ (unsigned)seed;
    }
    __syncthreads();
    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);

    unsigned s0=0, s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        int base = (tid * 4 + i * 128) & BOUND_MASK;
        unsigned r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r0) : "r"(local_base + base + 0));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r1) : "r"(local_base + base + 32));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r2) : "r"(local_base + base + 64));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r3) : "r"(local_base + base + 96));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r4) : "r"(local_base + base + 128));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r5) : "r"(local_base + base + 160));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r6) : "r"(local_base + base + 192));
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r7) : "r"(local_base + base + 224));
        s0 += r0; s1 += r1; s2 += r2; s3 += r3;
        s4 += r4; s5 += r5; s6 += r6; s7 += r7;
    }
    unsigned sum = s0^s1^s2^s3^s4^s5^s6^s7;
    out[gtid] = sum;
}

int main() {
    cudaSetDevice(0);
    unsigned int* d_out;
    CK(cudaMalloc(&d_out, N_BLOCKS * 128 * sizeof(unsigned int)));

    int ITERS = 50000;
    int seed = 42;

    // Configure cluster launch
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(N_BLOCKS, 1, 1);
    cfg.blockDim = dim3(128, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = 0;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_X;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    // Warmup
    for (int i = 0; i < 3; i++) {
        CK(cudaLaunchKernelEx(&cfg, kernel_dsmem, d_out, ITERS, seed));
    }
    CK(cudaDeviceSynchronize());

    // Time DSMEM
    cudaEvent_t e1, e2;
    cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventRecord(e1);
    for (int i = 0; i < 10; i++) {
        CK(cudaLaunchKernelEx(&cfg, kernel_dsmem, d_out, ITERS, seed));
    }
    cudaEventRecord(e2);
    CK(cudaEventSynchronize(e2));
    float ms_dsmem = 0;
    cudaEventElapsedTime(&ms_dsmem, e1, e2);
    ms_dsmem /= 10;

    // Time local SMEM baseline
    cudaEventRecord(e1);
    for (int i = 0; i < 10; i++) {
        kernel_local<<<N_BLOCKS, 128>>>(d_out, ITERS, seed);
        CK(cudaGetLastError());
    }
    cudaEventRecord(e2);
    CK(cudaEventSynchronize(e2));
    float ms_local = 0;
    cudaEventElapsedTime(&ms_local, e1, e2);
    ms_local /= 10;

    // Bytes: each thread × ITERS × 8 loads × 4 B
    double total_B = (double)N_BLOCKS * 128 * ITERS * 8 * 4;
    double gb = total_B / 1e9;
    double tb_dsmem = gb / (ms_dsmem / 1000.0) / 1e3;
    double tb_local = gb / (ms_local / 1000.0) / 1e3;

    printf("=== Cluster DSMEM vs Local SMEM BW ===\n");
    printf("  Config: %d blocks × 128 thr × %d iters × 8 LDS/iter\n", N_BLOCKS, ITERS);
    printf("  Total bytes: %.2f GB\n", gb);
    printf("  LOCAL  SMEM: %.2f ms → %.2f TB/s\n", ms_local, tb_local);
    printf("  DSMEM  (cluster=%d): %.2f ms → %.2f TB/s\n", CLUSTER_X, ms_dsmem, tb_dsmem);
    printf("  ratio DSMEM/local: %.2f×\n", tb_dsmem / tb_local);

    cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaFree(d_out);
    return 0;
}
