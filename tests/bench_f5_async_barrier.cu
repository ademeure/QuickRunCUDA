// F5: cp.async + mbarrier pipeline depth
// How many concurrent cp.async transactions can be in flight before stall?
// MODE 0: 1 cp.async per iter, 1 wait
// MODE 1: 2 cp.async per iter
// MODE 2: 4 cp.async per iter
// MODE 3: 8 cp.async per iter
// MODE 4: 16 cp.async per iter

#ifndef MODE
#define MODE 0
#endif

#define N_CHUNKS_PER_ITER (1 << MODE)
#define CHUNK_SIZE 16  // 16 bytes per cp.async

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ unsigned long long bar_storage[1];
    __shared__ __align__(16) unsigned int smem[4096];  // 16 KB, 16B aligned

    unsigned int bar_addr = __cvta_generic_to_shared(bar_storage);
    unsigned int smem_addr = __cvta_generic_to_shared(smem);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" :: "r"(bar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncwarp();

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Each thread loads 1 uint4 (16 B) per chunk. tid increments by 16 B = 4 floats.
        // src offset in floats = (i_chunk * 32 + c * 32 + tid) * 4 → byte offset is mult of 16.
        unsigned long long src_chunk = ((unsigned long long)(i ^ u2)) & ((1u<<18) - 1u);
        #pragma unroll
        for (int c = 0; c < N_CHUNKS_PER_ITER; c++) {
            unsigned int dst = smem_addr + (c * 32 + threadIdx.x) * 16;
            unsigned long long src_float_idx = ((src_chunk * 32) + (c * 32) + threadIdx.x) * 4;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(A + src_float_idx));
        }
        // Commit and wait
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Sentinel
    if ((int)smem[threadIdx.x] == seed) C[blockIdx.x] = (float)smem[threadIdx.x];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_CHUNKS=%d clk=%llu cy/iter=%.3f cy/chunk=%.3f\n",
               MODE, N_CHUNKS_PER_ITER, t1-t0,
               (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/(double)N_CHUNKS_PER_ITER);
    }
}
