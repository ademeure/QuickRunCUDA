// Test: can the compiler/runtime reclaim the reserved 1 KiB
// when the kernel doesn't use cluster/mbarrier/TMA features?
#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

namespace cg = cooperative_groups;

// Test 1: kernel that uses 232,448 (= opt-in max)
extern "C" __global__ void k_max_optin() {
    __shared__ char buf[232448];
    int tid = threadIdx.x;
    buf[tid * 1000] = (char)tid;  // sparse write
    __syncthreads();
    if (tid == 0) printf("max_optin: buf[0]=%d, addr=%p\n", buf[0], buf);
}

// Test 2: kernel attempting MORE than opt-in fails to compile
// extern "C" __global__ void k_above_optin() {
//     __shared__ char buf[233472];  // ptxas: "uses too much shared data (0x39000 bytes, 0x38c00 max)"
// }

// Test 3: kernel with cluster + max shmem
extern "C" __global__ void __cluster_dims__(2,1,1) k_cluster_max() {
    __shared__ char buf[230400];  // leave some room for cluster
    auto cluster = cg::this_cluster();
    cluster.sync();
    int tid = threadIdx.x;
    buf[tid * 1000] = (char)tid;
    if (tid == 0) printf("cluster_max: buf[0]=%d\n", buf[0]);
}

// Test 4: simple kernel — does compiler use offset 0?
extern "C" __global__ void k_simple_max() {
    __shared__ char buf[232448];  // max user
    int tid = threadIdx.x;
    buf[tid * 1000] = (char)tid;
    if (tid == 0) printf("simple_max: addr=%p (offset=%lld)\n", buf, (long long)buf & 0xFFFFFF);
}

// Test 5: dynamic shmem sized to 232448
extern "C" __global__ void k_dyn_max() {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    buf[tid * 1000] = (char)tid;
    if (tid == 0) printf("dyn_max: addr=%p (offset=%lld)\n", buf, (long long)buf & 0xFFFFFF);
}

// Test 6: dynamic shmem MORE than opt-in - what's the limit?
int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("# B300 reserved-shmem reclamation test\n");
    printf("# reservedSharedMemPerBlock = %zu, opt-in max = %zu, per-SM = %zu\n\n",
           prop.reservedSharedMemPerBlock, prop.sharedMemPerBlockOptin,
           prop.sharedMemPerMultiprocessor);

    // ===== Static decl tests =====
    cudaFuncAttributes fa;

    cudaFuncGetAttributes(&fa, (void*)k_max_optin);
    printf("k_max_optin (declares 232448 static):\n");
    printf("  sharedSizeBytes=%zu, maxDynShmem=%d\n", fa.sharedSizeBytes, fa.maxDynamicSharedSizeBytes);

    printf("k_above_optin: ptxas COMPILE FAIL (0x39000 bytes > 0x38c00 max)\n");
    printf("  → 232448 = HARD CEILING enforced by ptxas\n");

    cudaFuncGetAttributes(&fa, (void*)k_cluster_max);
    printf("k_cluster_max (declares 230400 + cluster):\n");
    printf("  sharedSizeBytes=%zu, maxDynShmem=%d\n", fa.sharedSizeBytes, fa.maxDynamicSharedSizeBytes);

    cudaFuncGetAttributes(&fa, (void*)k_dyn_max);
    printf("k_dyn_max:\n");
    printf("  sharedSizeBytes=%zu, maxDynShmem=%d\n", fa.sharedSizeBytes, fa.maxDynamicSharedSizeBytes);

    // ===== Try setting opt-in shmem to various sizes =====
    printf("\n# cudaFuncSetAttribute(MaxDynamicSharedMemorySize, X) for k_dyn_max:\n");
    int sizes[] = {49152, 100000, 200000, 232448, 232449, 233472, 233473, 250000};
    for (int sz : sizes) {
        cudaError_t r = cudaFuncSetAttribute((void*)k_dyn_max,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, sz);
        printf("  set %-7d : %s\n", sz, r == cudaSuccess ? "OK" : cudaGetErrorString(r));
    }

    // ===== Launch tests =====
    printf("\n# Launch tests (1 block, 1 thread):\n");

    cudaError_t r;

    // k_max_optin needs opt-in
    cudaFuncSetAttribute((void*)k_max_optin, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
    k_max_optin<<<1, 1>>>();
    cudaDeviceSynchronize();
    r = cudaGetLastError();
    printf("k_max_optin: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

    // k_above_optin would fail - skipped (compile time)

    // k_simple_max
    cudaFuncSetAttribute((void*)k_simple_max, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
    k_simple_max<<<1, 1>>>();
    cudaDeviceSynchronize();
    r = cudaGetLastError();
    printf("k_simple_max: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

    // k_dyn_max with various dynamic sizes
    for (int dyn : {49152, 100000, 200000, 232448, 233472}) {
        cudaFuncSetAttribute((void*)k_dyn_max, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
        k_dyn_max<<<1, 1, dyn>>>();
        cudaDeviceSynchronize();
        r = cudaGetLastError();
        printf("k_dyn_max with dyn=%-7d: %s\n", dyn, r == cudaSuccess ? "OK" : cudaGetErrorString(r));
    }

    return 0;
}
