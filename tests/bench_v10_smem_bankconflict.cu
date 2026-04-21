// V10: SMEM bank conflict cost precisely
// 32 banks. Stride S between lane accesses determines conflict:
//   S=1 (contiguous): no conflict
//   S=32 (same bank): 32-way conflict = 32× slowdown
//   S=2: 2-way? actually S=2 with 32 threads → threads 0,1,2..31 → banks 0,2,4..62%32 = 0,2,4..30,0,2..30
//     = 2 threads per bank = 2-way
#ifndef STRIDE
#define STRIDE 1
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    __shared__ unsigned int smem[1024];
    if (threadIdx.x >= 32) return;
    int lane = threadIdx.x;

    // Init SMEM
    for (int i = lane; i < 1024; i += 32) smem[i] = i;
    __syncthreads();

    // Init SMEM[i] with (i + stride_per_lane) so each load chains
    // lane reads position `pos`, gets next_pos from smem[pos]
    for (int i = lane; i < 1024; i += 32) {
        // Each lane's chain: pos_i → pos_{i+1} = (pos_i + stride) & 1023
        smem[i] = (unsigned)((i + STRIDE) & 1023);
    }
    __syncwarp();

    unsigned int v = lane * STRIDE & 1023;
    unsigned long long t0, t1;
    if (lane == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    __syncwarp();

    unsigned local_base = (unsigned)__cvta_generic_to_shared(smem);

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
        // True chain: next index comes from smem load
        asm volatile("ld.shared.u32 %0, [%1];"
                     : "=r"(v) : "r"(local_base + v * 4));
    }

    __syncwarp();
    if (lane == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned int*)C)[2] = v;
    }
}
