// FFMA register port pressure — does FFMA pipe have read-port limit?
// Mode 0: 1 unique source (fma a,a,a,a)
// Mode 1: 2 unique sources (fma a,b,a,b)
// Mode 2: 3 unique sources (fma a,b,c,a)
// Mode 3: 4 unique sources (fma a,b,c,d) - 4 distinct reads (chain in d)

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef PORT_MODE
#define PORT_MODE 2
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv[N_CHAINS], fb[N_CHAINS], fc[N_CHAINS], fd[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        fv[k] = (float)(threadIdx.x + k);
        fb[k] = (float)(threadIdx.x * 2 + k);
        fc[k] = (float)(threadIdx.x * 3 + k);
        fd[k] = (float)(threadIdx.x * 5 + k);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if PORT_MODE == 0
                // 1 unique: a*a+a (chain in a, reads only a)
                asm volatile("fma.rn.f32 %0, %0, %0, %0;"
                             : "+f"(fv[k]));
#elif PORT_MODE == 1
                // 2 unique: a*b+a (chain in a)
                asm volatile("fma.rn.f32 %0, %0, %1, %0;"
                             : "+f"(fv[k]) : "f"(fb[k]));
#elif PORT_MODE == 2
                // 3 unique: a*b+c (chain in a)
                asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                             : "+f"(fv[k]) : "f"(fb[k]), "f"(fc[k]));
#elif PORT_MODE == 3
                // 4 unique: chain in d, reads a,b,c,d (3 sources + dest)
                asm volatile("fma.rn.f32 %3, %0, %1, %2;"
                             : "+f"(fd[k]) : "f"(fb[k]), "f"(fc[k]), "f"(fv[k]));
#endif
            }
        }
    }

    float facc = 0.0f;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) facc += fv[k] + fd[k];
    if ((int)facc == seed)
        ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)facc;
}
