// D2: L1 associativity test via pointer-chase with controlled stride
// Walk through N entries spaced STRIDE bytes apart
// If N*STRIDE fits in L1 capacity, latency = L1 hit
// If aliased into too few sets (N entries hit same set), latency = L2 fetch
//
// L1 on B300 typically 128KB+ data + 16KB ICACHE; ways unknown
// We test stride values that are powers of 2, then sizes growing
#ifndef N_ELEMS
#define N_ELEMS 32
#endif
#ifndef STRIDE_LOG2
#define STRIDE_LOG2 8
#endif
#define STRIDE_BYTES (1u << STRIDE_LOG2)

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // Setup pointer chain: A[i] = ((i+1) % N_ELEMS) * STRIDE_BYTES / 4 (in float index)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (unsigned int i = 0; i < N_ELEMS; i++) {
            unsigned int next_byte_off = ((i + 1) % N_ELEMS) * STRIDE_BYTES;
            ((unsigned int*)A)[(i * STRIDE_BYTES) / 4] = next_byte_off / 4;
        }
    }
    __syncthreads();

    // Each thread does pointer chase (via global memory)
    unsigned int idx = (threadIdx.x % N_ELEMS) * STRIDE_BYTES / 4;
    if ((unsigned)u2 != 0xDEADBEEFu) idx = 0;  // ensure runtime dependency

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        idx = ((unsigned int*)A)[idx];
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (idx == (unsigned)seed) ((unsigned*)C)[blockIdx.x * 32 + threadIdx.x] = idx;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("N=%d STRIDE=%u total=%u B clk=%llu cy/load=%.2f\n",
               N_ELEMS, STRIDE_BYTES, N_ELEMS * STRIDE_BYTES,
               t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
