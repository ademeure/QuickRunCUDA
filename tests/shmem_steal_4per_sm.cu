// Force 4 blocks per SM and verify the 'steal reserved' trick works
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

extern "C" __global__ void k_steal_reserved_persistent(int *sm_assignment, int *check) {
    extern __shared__ char buf[];  // compiler-aware = 56 KB
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (tid == 0) sm_assignment[bid] = smid;

    // Touch entire 56 KiB compiler-aware area
    for (int i = tid; i < (56 * 1024); i += blockDim.x)
        buf[i] = (char)((tid + bid + 0x10) & 0xFF);

    // Steal: write a pattern to the reserved 1 KiB
    unsigned int pattern = 0xDEAD0000 | bid;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        asm volatile("st.shared.u32 [%0], %1;" :: "r"(offset), "r"(pattern + i) : "memory");
    }

    __syncthreads();

    // Read back stolen and verify
    int corruption = 0;
    for (int i = tid; i < 256; i += blockDim.x) {
        unsigned int offset = i * 4;
        unsigned int val;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(offset));
        if (val != pattern + i) corruption = 1;
    }

    // Read back compiler-aware and verify
    for (int i = tid; i < (56 * 1024); i += blockDim.x) {
        if (buf[i] != (char)((tid + bid + 0x10) & 0xFF)) corruption = 1;
    }

    // Atomic OR to gather corruption info
    if (tid == 0) atomicOr(&check[bid], corruption);
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int sm_count = prop.multiProcessorCount;

    int *d_sm, *d_check;
    int n_blocks = 4 * sm_count;  // = 592 to force 4 per SM
    cudaMalloc(&d_sm, n_blocks * sizeof(int));
    cudaMalloc(&d_check, n_blocks * sizeof(int));
    cudaMemset(d_check, 0, n_blocks * sizeof(int));

    cudaFuncSetAttribute((void*)k_steal_reserved_persistent,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, 56*1024);

    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                                                   (void*)k_steal_reserved_persistent, 256, 56*1024);

    printf("# B300 'steal reserved' verify - 4 blocks per SM\n");
    printf("# Compiler-aware: 56 KiB. Runtime touches full 57 KiB.\n");
    printf("# Total blocks launched: %d (= 4 × %d SMs)\n", n_blocks, sm_count);
    printf("# Occupancy says: max %d blocks/SM at 56 KB shmem\n\n", max_blocks_per_sm);

    k_steal_reserved_persistent<<<n_blocks, 256, 56*1024>>>(d_sm, d_check);
    cudaDeviceSynchronize();
    cudaError_t r = cudaGetLastError();
    printf("Launch result: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

    int *sm_arr = new int[n_blocks];
    int *check_arr = new int[n_blocks];
    cudaMemcpy(sm_arr, d_sm, n_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(check_arr, d_check, n_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Count blocks per SM
    int blocks_per_sm[200] = {0};
    int max_count = 0, total_corrupt = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (sm_arr[i] >= 0 && sm_arr[i] < 200) blocks_per_sm[sm_arr[i]]++;
        if (check_arr[i]) total_corrupt++;
    }
    for (int i = 0; i < 200; i++) {
        if (blocks_per_sm[i] > max_count) max_count = blocks_per_sm[i];
    }

    printf("\nMax blocks per SM observed: %d\n", max_count);
    printf("Distribution of blocks per SM:\n");
    int dist[10] = {0};
    for (int i = 0; i < 200; i++) {
        if (blocks_per_sm[i] > 0 && blocks_per_sm[i] < 10) dist[blocks_per_sm[i]]++;
    }
    for (int i = 1; i < 10; i++) {
        if (dist[i] > 0) printf("  %d blocks/SM: %d SMs\n", i, dist[i]);
    }

    printf("\nData corruption: %d / %d blocks corrupted\n", total_corrupt, n_blocks);
    if (total_corrupt == 0)
        printf("✓ TRICK WORKS: 4 blocks × 57 KiB co-resident on SAME SM, no corruption!\n");
    else
        printf("✗ TRICK FAILED: data corruption detected\n");

    delete[] sm_arr;
    delete[] check_arr;
    cudaFree(d_sm); cudaFree(d_check);
    return 0;
}
