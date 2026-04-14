// Deep probe into pipe_adu and pipe_uniform.

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

#if OP == 0   // bar.sync 0 (CTA barrier)
                asm volatile("bar.sync 0;");
#elif OP == 1 // barrier.sync 0 (newer form)
                asm volatile("barrier.sync 0;");
#elif OP == 2 // bar.arrive (async)
                asm volatile("bar.arrive 0, %0;" :: "r"(BLOCK_SIZE));
#elif OP == 3 // bar.red.popc (count)
                int cnt;
                asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0; bar.red.popc.u32 %0, 0, p; }" : "=r"(cnt) : "r"(x));
                x = (unsigned)cnt;
#elif OP == 4 // membar.cta
                asm volatile("membar.cta;");
#elif OP == 5 // membar.gl
                asm volatile("membar.gl;");
#elif OP == 6 // match.any (all threads give same value — less contention)
                asm volatile("match.any.sync.b32 %0, %0, 0xFFFFFFFF;" : "+r"(x));
#elif OP == 7 // match.all
                int pred;
                asm volatile("{ .reg .pred p; match.all.sync.b32 %0|p, %0, 0xFFFFFFFF; selp.u32 %0, 1, 2, p; }" : "+r"(x));
#elif OP == 8 // atom.shared.add (LSU pipe probably, test)
                asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(x) : "r"((unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x & 0x7) * 4])), "r"(1u));
#elif OP == 9 // LDSM .x1
                asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                    : "=r"(x) : "r"((unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x&0x1F)*8])));
#elif OP == 10 // LDSM .x2
                unsigned int a, b;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                    : "=r"(a), "=r"(b) : "r"((unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x&0x1F)*8])));
                x = a ^ b;
#elif OP == 11 // LDSM .x4
                unsigned int a, b, c, d;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                    : "r"((unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x&0x1F)*8])));
                x = a ^ b ^ c ^ d;
#elif OP == 12 // LDSM .x4 .trans
                unsigned int a, b, c, d;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                    : "r"((unsigned)__cvta_generic_to_shared(&smem[(threadIdx.x&0x1F)*8])));
                x = a ^ b ^ c ^ d;
#elif OP == 13 // activemask (uniform)
                asm volatile("activemask.b32 %0;" : "=r"(x));
#elif OP == 14 // S2R SR_LANEID
                asm volatile("mov.u32 %0, %%laneid;" : "=r"(x));
#elif OP == 15 // S2R SR_CLOCKLO
                asm volatile("mov.u32 %0, %%clock;" : "=r"(x));
#elif OP == 16 // S2R SR_WARPID
                asm volatile("mov.u32 %0, %%warpid;" : "=r"(x));
#elif OP == 17 // DEPBAR (dependency barrier) — via ptx
                asm volatile("bar.cta.sync.aligned 0;");
#elif OP == 18 // simple setp with uniform — S2UR pipe maybe
                // This forces UISETP-like pattern
                asm volatile("{ .reg .pred p; setp.ne.u32 p, %0, 0; selp.u32 %0, 1, 2, p; }" : "+r"(x));
#elif OP == 19 // redux.sync.add (warp reduce — new on sm_80+)
                asm volatile("redux.sync.add.u32 %0, %0, 0xFFFFFFFF;" : "+r"(x));
#elif OP == 20 // redux.sync.min
                asm volatile("redux.sync.min.u32 %0, %0, 0xFFFFFFFF;" : "+r"(x));
#elif OP == 21 // redux.sync.or
                asm volatile("redux.sync.or.b32 %0, %0, 0xFFFFFFFF;" : "+r"(x));
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
