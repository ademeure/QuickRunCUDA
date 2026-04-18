// C5: cp.async.cg vs cp.async.ca semantics for SMEM streaming
//
// Variants for GMEM → SMEM:
//   .cg = cache-global (bypass L1, save L1 for other uses)
//   .ca = cache-all (cache in L1 too)
//
// For SHMEM → GMEM via cp.async.bulk (TMA): no cache hint needed (always L2)
//
// Test: which gives best end-to-end BW for streaming workloads?
#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 128;

__device__ __forceinline__ void cp_async_cg_16(unsigned smem_addr, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;\n" :: "r"(smem_addr), "l"(src));
}
__device__ __forceinline__ void cp_async_ca_16(unsigned smem_addr, const void *src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 16;\n" :: "r"(smem_addr), "l"(src));
}
__device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

template <bool USE_CG>
__launch_bounds__(256, 4) __global__ void k_async_load(const uint4 *src, int *out, size_t N_uint4) {
    extern __shared__ uint4 smem[];
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned smem_addr_base = __cvta_generic_to_shared(smem);
    int4 acc = make_int4(0,0,0,0);
    for (size_t i = tid; i < N_uint4; i += stride) {
        unsigned dst_addr = smem_addr_base + (threadIdx.x * 16);
        if (USE_CG) cp_async_cg_16(dst_addr, &src[i]);
        else        cp_async_ca_16(dst_addr, &src[i]);
        commit_group();
        wait_all();
        uint4 v = smem[threadIdx.x];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xDEADBEEF && N_uint4 == 0)
        out[threadIdx.x] = acc.x;
}

// Plain LDG.128 baseline
__launch_bounds__(256, 8) __global__ void k_ldg(const uint4 *src, int *out, size_t N_uint4) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    int4 acc = make_int4(0,0,0,0);
    for (size_t i = tid; i < N_uint4; i += stride) {
        uint4 v = src[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xDEADBEEF && N_uint4 == 0)
        out[threadIdx.x] = acc.x;
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;
    uint4 *d_data; cudaMalloc(&d_data, bytes);
    cudaMemset(d_data, 0, bytes);
    size_t N_uint4 = bytes / 16;
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](const char* name, auto launch, double total_bytes) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s: %s\n", name, cudaGetErrorString(cudaGetLastError())); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double tbs = total_bytes / (best/1000.0) / 1e12;
        printf("  %-30s  %.3f ms  %.2f TB/s (%.0f%% HBM)\n", name, best, tbs, tbs/7.31*100);
    };

    int blocks = 148 * 4, threads = 256;
    int smem = 256 * 16;  // 4 KB per block
    cudaFuncSetAttribute(k_async_load<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16 * 1024);
    cudaFuncSetAttribute(k_async_load<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16 * 1024);

    printf("# cp.async variants for streaming GMEM→SMEM (N=%.0f MB total)\n", (double)bytes/1e6);
    bench("LDG.128 baseline (sync)", [&]() { k_ldg<<<blocks, threads>>>(d_data, d_out, N_uint4); }, bytes);
    bench("cp.async.cg (bypass L1)", [&]() { k_async_load<true><<<blocks, threads, smem>>>(d_data, d_out, N_uint4); }, bytes);
    bench("cp.async.ca (cache L1)",  [&]() { k_async_load<false><<<blocks, threads, smem>>>(d_data, d_out, N_uint4); }, bytes);
    return 0;
}
