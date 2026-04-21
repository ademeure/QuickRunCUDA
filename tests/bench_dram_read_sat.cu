// Sustained DRAM read bandwidth test (full chip)
#ifndef MODE
#define MODE 0
#endif
#ifndef ELEMS_PER_THREAD
#define ELEMS_PER_THREAD 4
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* p = (unsigned int*)A;
    unsigned int acc = 0;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Scalar 32-bit reads
        int idx = (gid + i * total_threads) & 0x3FFFFFF;
        acc ^= p[idx];
#elif MODE == 1
        // uint4 vectorized reads
        uint4* p4 = (uint4*)p;
        int idx = (gid + i * total_threads) & 0xFFFFFF;
        uint4 v = p4[idx];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
#elif MODE == 2
        // 4 unrolled uint4 reads (more in-flight)
        uint4* p4 = (uint4*)p;
        int idx = (gid + i * total_threads) & 0xFFFFFC;
        uint4 v0 = p4[idx];
        uint4 v1 = p4[idx + 1];
        uint4 v2 = p4[idx + 2];
        uint4 v3 = p4[idx + 3];
        acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w;
        acc ^= v1.x ^ v1.y ^ v1.z ^ v1.w;
        acc ^= v2.x ^ v2.y ^ v2.z ^ v2.w;
        acc ^= v3.x ^ v3.y ^ v3.z ^ v3.w;
#endif
    }

    if (acc == (unsigned)seed) ((unsigned*)C)[gid] = acc;
}
