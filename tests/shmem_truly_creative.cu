// Truly exotic attempts to fit 4×57KiB on same SM
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define TARGET (57 * 1024)  // 58368 bytes target

// Approach 1: Use the 1 KiB "reserved" via raw PTX!
// Compiler doesn't know we touch offset 0..1023, so it counts our shmem as 56 KiB
// But at runtime we use the full 57 KiB by ALSO touching reserved space
extern "C" __global__ void k_steal_reserved() {
    extern __shared__ char buf[];
    int tid = threadIdx.x;

    // Use compiler-allocated buf (= 56 KiB starting at offset 1024)
    for (int i = tid; i < (56 * 1024); i += blockDim.x) {
        buf[i] = (char)(tid ^ blockIdx.x);
    }

    // Now ADDITIONALLY touch offsets 0..1023 (the "reserved" space) via raw PTX
    // This doesn't go through compiler-aware __shared__, so compiler doesn't account for it
    for (int i = tid; i < 1024; i += blockDim.x) {
        unsigned int offset = i;
        unsigned char val = (unsigned char)(tid ^ blockIdx.x ^ 0xAA);
        asm volatile("st.shared.u8 [%0], %1;" :: "r"(offset), "r"((unsigned int)val) : "memory");
    }

    __syncthreads();

    // Read back the full 57 KiB (56 from compiler + 1 from raw)
    int sum = 0;
    for (int i = tid; i < (56 * 1024); i += blockDim.x)
        sum += (int)buf[i];
    for (int i = tid; i < 1024; i += blockDim.x) {
        unsigned int offset = i;
        unsigned int val;
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(val) : "r"(offset));
        sum += (int)val;
    }
    if (tid == 0) {
        atomicAdd((int*)((long)buf + (56*1024 - 4)), sum);
    }
}

// Approach 2: Pure compiler shmem at exactly 56 KiB - this should fit 4 blocks
extern "C" __global__ void k_56k_real() {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    int target_bytes = 56 * 1024;
    for (int i = tid; i < target_bytes; i += blockDim.x) {
        buf[i] = (char)(tid ^ blockIdx.x);
    }
    __syncthreads();
    int sum = 0;
    for (int i = tid; i < target_bytes; i += blockDim.x)
        sum += (int)buf[i];
    if (tid == 0) {
        atomicAdd((int*)((long)buf + (target_bytes - 4)), sum);
    }
}

// Approach 3: cudaFuncCachePreferShared (legacy hint, might do something)
// Plus old PTX `setmaxnreg` for register reduction
extern "C" __global__ void __launch_bounds__(256, 4) k_bounds_4() {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    for (int i = tid; i < TARGET; i += blockDim.x) {
        buf[i] = (char)(tid ^ blockIdx.x);
    }
    __syncthreads();
    int sum = 0;
    for (int i = tid; i < TARGET; i += blockDim.x)
        sum += (int)buf[i];
    if (tid == 0) {
        atomicAdd((int*)((long)buf + (TARGET - 4)), sum);
    }
}

// Persistent kernel that holds blocks on SM forever (forces 4 to coreside)
extern "C" __global__ void k_persistent_test() {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // Touch all 57 KiB
    for (int i = tid; i < TARGET; i += blockDim.x)
        buf[i] = (char)(tid ^ bid);
    __syncthreads();

    // Get and report SM ID
    int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (tid == 0) printf("Block %d -> SM %d\n", bid, smid);
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));

    printf("# B300 EXOTIC tests for 4×57KiB on same SM\n\n");

    // ===== Approach 1: Steal reserved =====
    printf("## Approach 1: 'steal' reserved 1KiB via raw PTX (compiler sees only 56 KiB)\n");
    {
        // Set compiler-visible to 56 KB
        cudaError_t r = cudaFuncSetAttribute((void*)k_steal_reserved,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, 56 * 1024);
        printf("  Set MaxDynShmem=56KB: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));
        int blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)k_steal_reserved, 256, 56*1024);
        printf("  occupancy says: %d blocks/SM\n", blocks);

        // Launch with 56 KB dyn (56 KiB * 4 + 4 KiB reserved = 228 KiB → fits!)
        cudaGetLastError();
        k_steal_reserved<<<4, 256, 56*1024>>>();
        cudaDeviceSynchronize();
        r = cudaGetLastError();
        printf("  Launch <<<4,256, 56KB>>>: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));
        // BUT — this only USES 57 KiB at runtime. Does the runtime crash?
    }

    // ===== Approach 2: Persistent — verify they all on SAME SM =====
    printf("\n## Approach 2: persistent kernel reporting SM IDs (4-block test)\n");
    {
        cudaFuncSetAttribute((void*)k_persistent_test,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, TARGET);
        printf("  Launching 4 blocks of %d byte dyn (57 KiB each):\n", TARGET);
        // 4 blocks of 57 KiB will run on 4 SMs (since 4 don't fit on one)
        k_persistent_test<<<4, 256, TARGET>>>();
        cudaDeviceSynchronize();
        cudaError_t r = cudaGetLastError();
        printf("  Result: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));
    }

    // ===== Approach 3: 56KB version, verify ALL on SAME SM (4 blocks fit) =====
    printf("\n## Approach 3: 4 blocks of 56 KiB - do they all fit on one SM?\n");
    {
        cudaFuncSetAttribute((void*)k_56k_real,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 56*1024);
        // Use cudaOccupancyMaxPotentialBlockSize to get config
        int min_grid, block_sz;
        cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_sz, (void*)k_56k_real, 56*1024);
        printf("  Max potential blockSize: %d, min grid: %d\n", block_sz, min_grid);
        int blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)k_56k_real, 256, 56*1024);
        printf("  Max blocks/SM: %d\n", blocks);

        // To force all on SAME SM: launch 4 blocks but they'll go to different SMs
        // unless we use cluster_dims=4
    }

    // ===== Approach 4: cluster=4 to FORCE same-SM placement (or cluster placement) =====
    printf("\n## Approach 4: cluster=4 with 57 KiB - does cluster pool shmem?\n");
    {
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = 4;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;
        cudaLaunchConfig_t cfg = {dim3(4), dim3(256), TARGET, 0, &attr, 1};

        cudaFuncSetAttribute((void*)k_persistent_test,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, TARGET);
        void *args[] = {};
        cudaError_t r = cudaLaunchKernelExC(&cfg, (void*)k_persistent_test, args);
        cudaDeviceSynchronize();
        printf("  Cluster_dim=4, dyn=57KB: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));
    }

    return 0;
}
