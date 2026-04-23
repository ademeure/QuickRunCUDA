// Cluster launch overhead probe for §22m. Body is empty so we measure only
// the launch path. Cluster size set at NVRTC time via -H "#define CSIZE N".
//
// Verification: kernel writes %cluster_nctaid.x to C[0] (first thread of
// first block) so we can confirm at runtime that the cluster IS active.

#ifndef CSIZE
#define CSIZE 1
#endif

#if CSIZE == 1
extern "C" __global__ void kernel(float* A, float* B, float* C, int u0, int u1, int u2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int n;
        asm("mov.u32 %0, %%ctaid.x;" : "=r"(n));
        ((unsigned int*)C)[0] = 1u;
        ((unsigned int*)C)[1] = n;
    }
}
#else
extern "C" __global__ __cluster_dims__(CSIZE,1,1) void kernel(float* A, float* B, float* C, int u0, int u1, int u2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned int n;
        asm("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(n));
        ((unsigned int*)C)[0] = n;
    }
}
#endif
