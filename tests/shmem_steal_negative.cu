// Negative test: does 'steal reserved' break when we ALSO use cluster.sync?
#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Steal reserved + use cluster.sync()
extern "C" __global__ void __cluster_dims__(2,1,1) k_steal_with_cluster(int *check) {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    auto cluster = cg::this_cluster();

    // Steal reserved space (write known pattern)
    unsigned int magic = 0xCAFE0000 | bid;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(magic + i) : "memory");
    }

    // Use cluster.sync() — this might use the reserved space
    cluster.sync();

    // Read back the stolen space
    int corruption = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corruption++;
    }
    if (tid == 0) check[bid] = corruption;
}

// Steal reserved + use mbarrier (alternative test)
extern "C" __global__ void k_steal_with_mbar(int *check) {
    extern __shared__ char buf[];  // user shmem starts at offset 1024
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Steal reserved space (write pattern)
    unsigned int magic = 0xBEEF0000 | bid;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(magic + i) : "memory");
    }

    // Init an mbarrier in USER shmem (at offset 1024 = start of buf)
    if (tid == 0) {
        unsigned int mbar_offset = 1024;  // first byte of user shmem
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(mbar_offset), "r"((unsigned)blockDim.x));
    }
    __syncthreads();

    // Arrive on mbarrier (might use reserved space?)
    unsigned long long token;
    unsigned int mbar_offset = 1024;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
                 : "=l"(token) : "r"(mbar_offset));
    int waitsuc;
    asm volatile("{ .reg .pred p;\n\t mbarrier.test_wait.shared.b64 p, [%1], %2; selp.s32 %0, 1, 0, p; }"
                 : "=r"(waitsuc) : "r"(mbar_offset), "l"(token));

    // Read back stolen space
    int corruption = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corruption++;
    }
    if (tid == 0) check[bid] = corruption;
}

// Steal reserved + use PDL signal
extern "C" __global__ void k_steal_with_pdl(int *check) {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    unsigned int magic = 0x12340000 | bid;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(magic + i) : "memory");
    }

    // PDL signal
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    // Read back
    int corruption = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corruption++;
    }
    if (tid == 0) check[bid] = corruption;
}

int main() {
    cudaSetDevice(0);

    int *d_check;
    cudaMalloc(&d_check, 16 * sizeof(int));

    printf("# Test 1: 'steal reserved' + cluster.sync() — does cluster overwrite stolen?\n");
    {
        cudaMemset(d_check, 0, 16 * sizeof(int));
        cudaFuncSetAttribute((void*)k_steal_with_cluster,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 4096);
        k_steal_with_cluster<<<2, 32, 4096>>>(d_check);
        cudaDeviceSynchronize();
        cudaError_t r = cudaGetLastError();
        printf("  Launch: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

        int check[16];
        cudaMemcpy(check, d_check, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 2; i++)
            printf("  Block %d: %d corrupted words\n", i, check[i]);
    }

    printf("\n# Test 2: 'steal reserved' + mbarrier in USER shmem\n");
    {
        cudaMemset(d_check, 0, 16 * sizeof(int));
        cudaFuncSetAttribute((void*)k_steal_with_mbar,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 4096);
        k_steal_with_mbar<<<1, 32, 4096>>>(d_check);
        cudaDeviceSynchronize();
        cudaError_t r = cudaGetLastError();
        printf("  Launch: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

        int check[16];
        cudaMemcpy(check, d_check, 1 * sizeof(int), cudaMemcpyDeviceToHost);
        printf("  Block 0: %d corrupted words\n", check[0]);
    }

    printf("\n# Test 3: 'steal reserved' + PDL launch_dependents\n");
    {
        cudaMemset(d_check, 0, 16 * sizeof(int));
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {dim3(1), dim3(32), 4096, 0, &attr, 1};

        cudaFuncSetAttribute((void*)k_steal_with_pdl,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 4096);

        void *args[] = {&d_check};
        cudaLaunchKernelExC(&cfg, (void*)k_steal_with_pdl, args);
        cudaDeviceSynchronize();
        cudaError_t r = cudaGetLastError();
        printf("  Launch: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

        int check[16];
        cudaMemcpy(check, d_check, 1 * sizeof(int), cudaMemcpyDeviceToHost);
        printf("  Block 0: %d corrupted words\n", check[0]);
    }

    cudaFree(d_check);
    return 0;
}
