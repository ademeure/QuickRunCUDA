// Wider shared memory load pattern sweep
// v2.b32 = 8-byte/lane = spans 2 banks per lane → 64 banks needed per warp
// v4.b32 = 16-byte/lane = spans 4 banks per lane → 128 banks needed per warp
// B300 has 32 banks × 4B = 128B/cycle, so v4 needs to be perfectly aligned

#ifndef PATTERN
#define PATTERN 0
#endif
#ifndef WIDTH
#define WIDTH 1   // 1 = b32, 2 = v2.b32, 4 = v4.b32
#endif
#ifndef ITERS
#define ITERS 10000
#endif

#define SMEM_SIZE 4096

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    __shared__ int smem[SMEM_SIZE];
    int lane = threadIdx.x;

    for (int i = lane; i < SMEM_SIZE; i += 32) {
        smem[i] = i * 1103515245u + 12345u + (unsigned)seed;
    }
    __syncwarp();

    int idx_base;
    #if PATTERN == 0
        idx_base = lane;
    #elif PATTERN == 1
        idx_base = (lane * 2) & 31;
    #elif PATTERN == 2
        idx_base = (lane * 4) & 31;
    #elif PATTERN == 3
        idx_base = (lane * 8) & 31;
    #elif PATTERN == 4
        idx_base = (lane * 16) & 31;
    #elif PATTERN == 5
        idx_base = 0;  // broadcast
    #elif PATTERN == 6
        unsigned int r = (unsigned)(lane + 1);
        r ^= r << 13; r ^= r >> 17; r ^= r << 5;
        idx_base = r & 31;
    #endif

    int acc_x = 0, acc_y = 0, acc_z = 0, acc_w = 0;
    unsigned long long t0, t1;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 16
    for (int i = 0; i < ITERS; i++) {
        // idx spans different rows per iter
        int idx = ((i & 31) * 32) + idx_base;
        idx &= (SMEM_SIZE/WIDTH - 1);  // keep in bounds after widening
        #if WIDTH == 1
            acc_x ^= smem[idx];
        #elif WIDTH == 2
            int2 v = ((int2*)smem)[idx];
            acc_x ^= v.x; acc_y ^= v.y;
        #elif WIDTH == 4
            int4 v = ((int4*)smem)[idx];
            acc_x ^= v.x; acc_y ^= v.y; acc_z ^= v.z; acc_w ^= v.w;
        #endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if ((acc_x ^ acc_y ^ acc_z ^ acc_w) == 0xDEADBEEF) C[lane] = (float)acc_x;

    if (lane == 0) {
        ((unsigned long long*)C)[1024 + blockIdx.x] = t1 - t0;
    }
}
