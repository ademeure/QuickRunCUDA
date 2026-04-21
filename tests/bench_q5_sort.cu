// Q5: Bitonic sort 1024 keys in single block (256 threads, 4 keys each in SMEM)
// Bitonic = O(N log^2 N) compares; log2(1024)^2 = 100 compare-swaps per element
extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(16) unsigned int smem[1024];
    int tid = threadIdx.x;

    // Init with reverse-sorted values mixed with u2 to defeat compiler folding
    smem[tid * 4 + 0] = 1024 - tid * 4 + (unsigned)u2;
    smem[tid * 4 + 1] = 1024 - tid * 4 - 1 + (unsigned)u2;
    smem[tid * 4 + 2] = 1024 - tid * 4 - 2 + (unsigned)u2;
    smem[tid * 4 + 3] = 1024 - tid * 4 - 3 + (unsigned)u2;
    __syncthreads();

    unsigned long long t0, t1;
    if (tid == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < ITERS; it++) {
        // Bitonic sort over 1024 elements. 4 indices per thread.
        // Each thread holds 4 elements in registers, sorts them locally, then merges.
        // Standard bitonic: log2(N) outer × log2(N) inner = 10 × 10 = 100 stages.
        for (int k = 2; k <= 1024; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                // Each of 256 threads handles 4 element-pairs
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    int idx = tid * 4 + e;
                    int ixj = idx ^ j;
                    if (ixj > idx) {
                        unsigned int va = smem[idx];
                        unsigned int vb = smem[ixj];
                        bool ascending = ((idx & k) == 0);
                        if ((va > vb) == ascending) {
                            smem[idx] = vb;
                            smem[ixj] = va;
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        printf("Q5 sort 1024 keys: clk=%llu cy/sort=%.0f\n",
               t1-t0, (double)(t1-t0)/(double)ITERS);
    }
    if (smem[1023] == 0xDEADBEEF) C[blockIdx.x] = (float)smem[1023];
}
