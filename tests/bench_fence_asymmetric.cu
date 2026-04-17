// Asymmetric: some SMs fence, others just write heavy (no fences).
// Measure fence cost on fencer SMs under various writer-SM loads.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef FENCER_SMS
#define FENCER_SMS 16
#endif
#ifndef WRITER_WRITES
#define WRITER_WRITES 16
#endif
#ifndef FENCER_WARPS_PER_SM
#define FENCER_WARPS_PER_SM 1
#endif
#ifndef FENCER_WRITES
#define FENCER_WRITES 1
#endif
#ifndef ITERS
#define ITERS 200
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned* my_addr = A + tid * WRITER_WRITES;
    unsigned warp_id = threadIdx.x >> 5;
    unsigned lane = threadIdx.x & 31;
    bool is_fencer = (blockIdx.x < FENCER_SMS);

    if (!is_fencer) {
        // WRITER SMs: heavy write loop, no fences
        #pragma unroll 1
        for (int i = 0; i < 1000000; i++) {
            #pragma unroll
            for (int j = 0; j < WRITER_WRITES; j++) ((volatile unsigned*)my_addr)[j] = i + seed + j;
        }
        return;
    }

    // FENCER SMs: measure fence cost
    if (warp_id >= FENCER_WARPS_PER_SM) return;

    unsigned long long t0, t1;
    unsigned total = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
#if FENCER_WRITES > 0
        #pragma unroll
        for (int j = 0; j < FENCER_WRITES; j++) ((volatile unsigned*)my_addr)[j] = i + seed;
#endif
#if OP == 0
        asm volatile("fence.sc.sys;");
#elif OP == 1
        asm volatile("fence.sc.gpu;");
#endif
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total += (unsigned)(t1 - t0);
    }
    if (lane == 0) {
        C[blockIdx.x * FENCER_WARPS_PER_SM + warp_id] = total / ITERS;
    }
}
