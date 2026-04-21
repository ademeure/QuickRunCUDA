// LDS shared memory bank-conflict cost.
// 32 banks × 4 bytes. Stride pattern controls conflict count.
// MODE = stride in bytes (4=no conflict, 8=2-way, 16=4-way, 32=8-way, 128=32-way)
#ifndef STRIDE
#define STRIDE 4
#endif

extern "C" __global__ __launch_bounds__(32, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ unsigned int smem[2048];

    // init smem
    for (int i = threadIdx.x; i < 2048; i += 32) smem[i] = i + (unsigned)u2;
    __syncwarp();

    unsigned int idx = (threadIdx.x * (STRIDE/4)) & 1023;
    unsigned int v = 0;

    // Get smem base as a generic pointer-as-uint
    unsigned int base_addr = __cvta_generic_to_shared(smem);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned int off = ((idx + i) & 1023) * 4;
        unsigned int x;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(base_addr + off));
        v ^= x;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("STRIDE=%d clk=%llu cy/load=%.3f\n",
               STRIDE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
