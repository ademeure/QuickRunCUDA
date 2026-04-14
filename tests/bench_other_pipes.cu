// Probe load/store/shuffle/barrier/branch pipes.

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) v[k] = 0xDEADBEEFu ^ (threadIdx.x * 131 + k * 17);
    // Prime shared memory
    if (threadIdx.x < 2048) smem[threadIdx.x] = threadIdx.x * 37;
    __syncthreads();

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned int x = v[k];
                unsigned int nxt = v[(k+1) & (N_CHAINS-1)];
#if OP == 0  // SHFL.SYNC.BFLY
                asm volatile("shfl.sync.bfly.b32 %0, %0, %1, 0x1F, 0xFFFFFFFF;" : "+r"(x) : "r"(nxt & 0x1F));
#elif OP == 1 // SHFL.SYNC.IDX
                asm volatile("shfl.sync.idx.b32 %0, %0, %1, 0x1F, 0xFFFFFFFF;" : "+r"(x) : "r"(nxt & 0x1F));
#elif OP == 2 // SHFL.SYNC.UP
                asm volatile("shfl.sync.up.b32 %0, %0, %1, 0, 0xFFFFFFFF;" : "+r"(x) : "r"(nxt & 0x1F));
#elif OP == 3 // SHFL.SYNC.DOWN
                asm volatile("shfl.sync.down.b32 %0, %0, %1, 0x1F, 0xFFFFFFFF;" : "+r"(x) : "r"(nxt & 0x1F));
#elif OP == 4 // LDS (shared load)
                unsigned int addr = (threadIdx.x * 4 + k * 4) & 0x1FFC;
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"((unsigned)__cvta_generic_to_shared(&smem[addr/4])));
#elif OP == 5 // STS (shared store)
                unsigned int addr = ((threadIdx.x + k) * 4) & 0x1FFC;
                asm volatile("st.shared.u32 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(&smem[addr/4])), "r"(x));
#elif OP == 6 // LDG (global load — warning: bandwidth-bound)
                asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(x) : "l"((unsigned long long)(A + (threadIdx.x + k * 1024))));
#elif OP == 7 // STG (global store)
                asm volatile("st.global.u32 [%0], %1;" :: "l"((unsigned long long)(B + (threadIdx.x + k * 1024))), "r"(x));
#elif OP == 8 // BAR.SYNC
                asm volatile("bar.sync 0;");
#elif OP == 9 // VOTE.BALLOT
                asm volatile("{.reg .pred p; setp.ne.u32 p, %0, 0; vote.sync.ballot.b32 %0, p, 0xFFFFFFFF;}" : "+r"(x));
#elif OP == 10 // ACTIVEMASK
                asm volatile("activemask.b32 %0;" : "=r"(x));
#elif OP == 11 // MATCH.ANY
                asm volatile("match.any.sync.b32 %0, %0, 0xFFFFFFFF;" : "+r"(x));
#elif OP == 12 // Branch (BRA — needs runtime condition)
                if (x & 1) asm volatile("" : "+r"(x));
                else asm volatile("" : "+r"(nxt));
                x = x ^ nxt;
#elif OP == 13 // S2R (read SR_TID.X — per-thread identity)
                asm volatile("mov.u32 %0, %%tid.x;" : "=r"(x));
#elif OP == 14 // LDSM (shared-memory matrix load — tensor adjacent)
                asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(x) : "r"((unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x&0x7)*8])));
#endif
                v[k] = x;
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
