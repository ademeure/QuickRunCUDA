// Steal reserved trick at HIGH occupancy: 8, 16, 32 blocks/SM
#include <cuda_runtime.h>
#include <cstdio>

template<int N_THREADS, int SHMEM_SIZE>
__global__ void k_steal_test(int *check) {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Steal: write magic to reserved 256 words
    unsigned int magic = 0x10000 * bid;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(magic + i) : "memory");
    }
    // Use compiler-aware area
    for (int i = tid; i < SHMEM_SIZE; i += blockDim.x) {
        buf[i] = (char)((tid + bid) & 0xFF);
    }
    __syncthreads();

    // Verify both
    int corrupt = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corrupt++;
    }
    for (int i = tid; i < SHMEM_SIZE; i += blockDim.x) {
        if (buf[i] != (char)((tid + bid) & 0xFF)) corrupt++;
    }
    if (tid == 0) check[bid] = corrupt;
}

// Per-instance kernels (template instantiation requires this approach)
extern "C" __global__ void k_steal_28k(int *check) {
    extern __shared__ char buf[];
    int tid = threadIdx.x, bid = blockIdx.x;
    unsigned int magic = 0x10000 * bid;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(magic + i) : "memory");
    }
    for (int i = tid; i < 28*1024; i += blockDim.x) buf[i] = (char)((tid + bid) & 0xFF);
    __syncthreads();
    int corrupt = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corrupt++;
    }
    for (int i = tid; i < 28*1024; i += blockDim.x)
        if (buf[i] != (char)((tid + bid) & 0xFF)) corrupt++;
    if (tid == 0) check[bid] = corrupt;
}

extern "C" __global__ void k_steal_12k(int *check) {
    extern __shared__ char buf[];
    int tid = threadIdx.x, bid = blockIdx.x;
    unsigned int magic = 0x10000 * bid;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(magic + i) : "memory");
    }
    for (int i = tid; i < 12*1024; i += blockDim.x) buf[i] = (char)((tid + bid) & 0xFF);
    __syncthreads();
    int corrupt = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corrupt++;
    }
    for (int i = tid; i < 12*1024; i += blockDim.x)
        if (buf[i] != (char)((tid + bid) & 0xFF)) corrupt++;
    if (tid == 0) check[bid] = corrupt;
}

extern "C" __global__ void k_steal_4k(int *check) {
    extern __shared__ char buf[];
    int tid = threadIdx.x, bid = blockIdx.x;
    unsigned int magic = 0x10000 * bid;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(magic + i) : "memory");
    }
    for (int i = tid; i < 4*1024; i += blockDim.x) buf[i] = (char)((tid + bid) & 0xFF);
    __syncthreads();
    int corrupt = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != magic + i) corrupt++;
    }
    for (int i = tid; i < 4*1024; i += blockDim.x)
        if (buf[i] != (char)((tid + bid) & 0xFF)) corrupt++;
    if (tid == 0) check[bid] = corrupt;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;

    int *d_check;
    int total = 32 * sm_count;
    cudaMalloc(&d_check, total * sizeof(int));

    printf("# B300 'steal reserved' at high occupancy\n");
    printf("# Each test: launch enough blocks to force max blocks/SM, verify all blocks intact\n\n");

    auto run_test = [&](const char *name, void *fn, int n_threads, int shmem_kb, int target_blocks_per_sm) {
        int shmem_bytes = shmem_kb * 1024;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);
        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, fn, n_threads, shmem_bytes);

        int n_blocks = max_blocks * sm_count;
        cudaMemset(d_check, 0, total * sizeof(int));

        cudaError_t r;
        cudaLaunchKernel(fn, dim3(n_blocks), dim3(n_threads), nullptr, shmem_bytes, 0);
        r = cudaGetLastError();
        cudaDeviceSynchronize();
        cudaError_t r2 = cudaGetLastError();

        int *check = new int[n_blocks];
        cudaMemcpy(check, d_check, n_blocks * sizeof(int), cudaMemcpyDeviceToHost);

        int corrupt_blocks = 0;
        for (int i = 0; i < n_blocks; i++) if (check[i] > 0) corrupt_blocks++;

        printf("  %s (%d thr, %d KiB shmem, %d blocks/SM): launch %s, sync %s, corrupt blocks=%d/%d\n",
               name, n_threads, shmem_kb, max_blocks,
               r == cudaSuccess ? "OK" : cudaGetErrorString(r),
               r2 == cudaSuccess ? "OK" : cudaGetErrorString(r2),
               corrupt_blocks, n_blocks);
        delete[] check;
    };

    // Need to launch with kernel function pointers cast appropriately
    void *args1[] = {&d_check};
    auto run_test2 = [&](const char *name, void *fn, int n_threads, int shmem_kb) {
        int shmem_bytes = shmem_kb * 1024;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);
        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, fn, n_threads, shmem_bytes);

        int n_blocks = max_blocks * sm_count;
        cudaMemset(d_check, 0, total * sizeof(int));

        cudaError_t r;
        cudaLaunchKernel(fn, dim3(n_blocks), dim3(n_threads), args1, shmem_bytes, 0);
        r = cudaGetLastError();
        cudaDeviceSynchronize();
        cudaError_t r2 = cudaGetLastError();

        int *check = new int[n_blocks];
        cudaMemcpy(check, d_check, n_blocks * sizeof(int), cudaMemcpyDeviceToHost);

        int corrupt_blocks = 0;
        for (int i = 0; i < n_blocks; i++) if (check[i] > 0) corrupt_blocks++;

        printf("  %s (%d thr, %d KiB shmem, %d blocks/SM): launch %s, sync %s, corrupt blocks=%d/%d\n",
               name, n_threads, shmem_kb, max_blocks,
               r == cudaSuccess ? "OK" : cudaGetErrorString(r),
               r2 == cudaSuccess ? "OK" : cudaGetErrorString(r2),
               corrupt_blocks, n_blocks);
        delete[] check;
    };

    run_test2("28KB", (void*)k_steal_28k, 256, 28);  // 8 blocks/SM expected
    run_test2("12KB", (void*)k_steal_12k, 128, 12);  // 16 blocks/SM
    run_test2("4KB",  (void*)k_steal_4k, 64, 4);     // 32 blocks/SM

    cudaFree(d_check);
    return 0;
}
