// VERIFY the "steal reserved" trick: 4 blocks × 57 KiB on SAME SM
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define TARGET (57 * 1024)  // 58368 bytes target

// Approach 1: compiler sees 56 KB, we steal the 1 KB reserved at offset 0..1023
extern "C" __global__ void k_steal_reserved_verify(int *sm_ids, int *block_marker) {
    extern __shared__ char buf_at_1024[];  // compiler places at offset 1024 (after reserved)
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Get SM ID
    int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (tid == 0) sm_ids[bid] = smid;

    // Compiler-aware: write 56 KiB (offset 1024 .. 1024+57344-1)
    int compiler_aware_size = 56 * 1024;
    for (int i = tid; i < compiler_aware_size; i += blockDim.x) {
        buf_at_1024[i] = (char)((tid + bid) & 0xFF);
    }

    // Steal: write reserved 1 KiB (offset 0..1023) via raw PTX
    for (int i = tid; i < 1024; i += blockDim.x) {
        unsigned int offset = i;
        unsigned int val = (unsigned int)((tid + bid + 0x80) & 0xFF);
        asm volatile("st.shared.u8 [%0], %1;" :: "r"(offset), "r"(val) : "memory");
    }

    __syncthreads();

    // Verify: read back compiler-aware
    int sum = 0;
    for (int i = tid; i < compiler_aware_size; i += blockDim.x) {
        sum += (int)buf_at_1024[i];
    }
    // Verify: read back stolen
    for (int i = tid; i < 1024; i += blockDim.x) {
        unsigned int offset = i;
        unsigned int val;
        asm volatile("ld.shared.u8 %0, [%1];" : "=r"(val) : "r"(offset));
        sum += (int)val;
    }

    if (tid == 0) {
        block_marker[bid] = sum;
    }
}

// Compute persistent kernel that does 57 KB of work
extern "C" __global__ void k_persistent_57k(int *sm_ids, unsigned long long *clk_starts, unsigned long long *clk_ends) {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (tid == 0) {
        sm_ids[bid] = smid;
        unsigned long long c;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c));
        clk_starts[bid] = c;
    }

    // Touch all 57 KiB
    for (int i = tid; i < (56 * 1024); i += blockDim.x)
        buf[i] = (char)(tid ^ bid);
    for (int i = tid; i < 1024; i += blockDim.x) {
        unsigned int offset = i;
        unsigned int val = (unsigned int)((tid ^ bid ^ 0x80) & 0xFF);
        asm volatile("st.shared.u8 [%0], %1;" :: "r"(offset), "r"(val) : "memory");
    }

    __syncthreads();

    // Sit and spin to force coresidency
    for (int i = 0; i < 100000; i++) {
        asm volatile("nanosleep.u32 100;");
    }

    if (tid == 0) {
        unsigned long long c;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(c));
        clk_ends[bid] = c;
    }
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));

    printf("# B300 verify 'steal reserved' trick: 4×57KiB on SAME SM\n\n");

    int *d_sm_ids, *d_block_marker;
    cudaMalloc(&d_sm_ids, 16 * sizeof(int));
    cudaMalloc(&d_block_marker, 16 * sizeof(int));

    // Approach 1: launch with compiler-aware 56 KB
    cudaFuncSetAttribute((void*)k_steal_reserved_verify,
                          cudaFuncAttributeMaxDynamicSharedMemorySize, 56*1024);
    int blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)k_steal_reserved_verify, 256, 56*1024);
    printf("  Occupancy with 56KB compiler-aware: %d blocks/SM\n", blocks);

    // Run 4 blocks
    k_steal_reserved_verify<<<4, 256, 56*1024>>>(d_sm_ids, d_block_marker);
    cudaDeviceSynchronize();
    cudaError_t r = cudaGetLastError();
    printf("  Launch 4 blocks: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

    int sm_ids[16];
    cudaMemcpy(sm_ids, d_sm_ids, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    int markers[16];
    cudaMemcpy(markers, d_block_marker, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; i++)
        printf("  Block %d: SM=%d, marker=%d\n", i, sm_ids[i], markers[i]);

    int unique_sm = 1;
    for (int i = 1; i < 4; i++) if (sm_ids[i] != sm_ids[0]) unique_sm++;
    printf("\n  Distinct SMs used: %d %s\n", unique_sm, unique_sm == 1 ? "  ← ALL 4 BLOCKS ON SAME SM!!!" : "  (different SMs)");

    // Persistent kernel test for forced co-residency
    printf("\n# Persistent kernel test (4 blocks of 57 KiB, sleep to force coresidency)\n");
    {
        unsigned long long *d_starts, *d_ends;
        cudaMalloc(&d_starts, 16 * sizeof(unsigned long long));
        cudaMalloc(&d_ends, 16 * sizeof(unsigned long long));

        cudaFuncSetAttribute((void*)k_persistent_57k,
                              cudaFuncAttributeMaxDynamicSharedMemorySize, 56*1024);

        k_persistent_57k<<<4, 256, 56*1024>>>(d_sm_ids, d_starts, d_ends);
        cudaDeviceSynchronize();
        cudaError_t r = cudaGetLastError();
        printf("  Persistent kernel launch: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

        cudaMemcpy(sm_ids, d_sm_ids, 4 * sizeof(int), cudaMemcpyDeviceToHost);
        unsigned long long starts[16], ends[16];
        cudaMemcpy(starts, d_starts, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(ends, d_ends, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 4; i++) {
            printf("  Block %d: SM=%d, start=%llu, end=%llu (dur=%llu ns)\n",
                   i, sm_ids[i], starts[i], ends[i], ends[i] - starts[i]);
        }

        // Check overlap
        unsigned long long max_start = starts[0];
        unsigned long long min_end = ends[0];
        for (int i = 1; i < 4; i++) {
            if (starts[i] > max_start) max_start = starts[i];
            if (ends[i] < min_end) min_end = ends[i];
        }
        printf("\n  Overlap window: %lld ns (positive = co-resident)\n",
               (long long)min_end - (long long)max_start);

        unique_sm = 1;
        for (int i = 1; i < 4; i++) if (sm_ids[i] != sm_ids[0]) unique_sm++;
        printf("  Distinct SMs in persistent test: %d\n", unique_sm);

        cudaFree(d_starts); cudaFree(d_ends);
    }

    cudaFree(d_sm_ids); cudaFree(d_block_marker);
    return 0;
}
