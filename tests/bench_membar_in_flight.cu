// membar.sys with pending writes always in flight.
// Each SM has 1 CTA = 32 warps. Half the warps continuously write to global;
// the "observer" warp does membar.sys + measures time.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef ITERS
#define ITERS 100
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned warp_id = threadIdx.x >> 5;
    unsigned* my_addr = A + tid;

    // First half of warps: continuous writers (keep writes in flight)
    if (warp_id >= 16) {
        // Keep writing continuously
        #pragma unroll 1
        for (int i = 0; i < 1000000; i++) {
            *(volatile unsigned*)my_addr = i;
        }
        return;
    }

    // Observer warp (warp 0-15): measure membar.sys cost
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned acc = seed;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0
        // Just membar.sys — no writes from observer, but writes from writer warps in flight
        asm volatile("membar.sys;");
#elif OP == 1
        // Observer also writes 4 then membar.sys
        ((volatile unsigned*)my_addr)[0] = i;
        ((volatile unsigned*)my_addr)[1] = i+1;
        ((volatile unsigned*)my_addr)[2] = i+2;
        ((volatile unsigned*)my_addr)[3] = i+3;
        asm volatile("membar.sys;");
#elif OP == 2
        // membar.gl instead of sys
        asm volatile("membar.gl;");
#elif OP == 3
        // membar.gl + 4 writes
        ((volatile unsigned*)my_addr)[0] = i;
        ((volatile unsigned*)my_addr)[1] = i+1;
        ((volatile unsigned*)my_addr)[2] = i+2;
        ((volatile unsigned*)my_addr)[3] = i+3;
        asm volatile("membar.gl;");
#endif
        acc ^= i;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
    }
}
