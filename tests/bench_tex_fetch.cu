// Texture fetch comparison vs LDG (texture mostly obsolete on B300?)
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Just test if tex sampler intrinsics work / are available
    // Texture API requires explicit setup; we'll fall back to __ldg path
    unsigned int* p = (unsigned int*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    unsigned int v;

    // __ldg is the modern equivalent of texture for read-only data
    v = __ldg(p + idx);

    if (v == (unsigned)seed) ((unsigned*)C)[idx] = v;
}
