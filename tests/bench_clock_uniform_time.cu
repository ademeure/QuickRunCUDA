#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    unsigned acc = 0;
    #pragma unroll 1
    for (int i = 0; i < 1024; i++) {
#if OP == 0
        // tid==0 + u32
        unsigned t0, t1;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
        if (threadIdx.x == 0) acc += t1 - t0;
#elif OP == 1
        // tid==0 + u64
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        if (threadIdx.x == 0) acc += (unsigned)(t1 - t0);
#elif OP == 2
        // laneid==0 + u32
        unsigned t0, t1;
        unsigned lane;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
        if (lane == 0) acc += t1 - t0;
#elif OP == 3
        // laneid==0 + u64
        unsigned long long t0, t1;
        unsigned lane;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
        if (lane == 0) acc += (unsigned)(t1 - t0);
#endif
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = end - start;
        C[2] = acc;
    }
}
