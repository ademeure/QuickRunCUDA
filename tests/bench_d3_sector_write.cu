// D3: L2 sector vs line write granularity — read amplification test
// Each thread writes WSIZE bytes per stride STRIDE bytes
// MODE 0: 4B writes spaced 32B apart (sub-sector → potential RMW)
// MODE 1: 4B writes spaced 128B apart (one per L2 line)
// MODE 2: 16B writes spaced 32B apart (half-sector)
// MODE 3: 32B writes spaced 32B apart (full sector, contiguous)
// MODE 4: 32B writes spaced 128B apart (one per line)
// MODE 5: 128B writes spaced 128B apart (full line)
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
#define WSIZE 4
#define STRIDE 32
#elif MODE == 1
#define WSIZE 4
#define STRIDE 128
#elif MODE == 2
#define WSIZE 16
#define STRIDE 32
#elif MODE == 3
#define WSIZE 32
#define STRIDE 32
#elif MODE == 4
#define WSIZE 32
#define STRIDE 128
#elif MODE == 5
#define WSIZE 128
#define STRIDE 128
#endif

extern "C" __global__ __launch_bounds__(128, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;
    if (gtid == 0 && blockIdx.x == 0) {
        // Sentinel
        C[0] = (float)WSIZE;
    }

    // Each thread writes WSIZE bytes per its STRIDE-byte stride
    // Total memory: total_threads * STRIDE bytes
    char* base = (char*)C;

    // Wrap to keep within buffer (256 MB = 1<<28)
    unsigned int wrap_mask = (1u << 28) - 1u - (STRIDE-1);
    for (int i = 0; i < ITERS; i++) {
        unsigned int off = ((gtid + i * total_threads) * STRIDE) & wrap_mask;
#if WSIZE == 4
        *((unsigned int*)(base + off)) = (unsigned)(i + u2);
#elif WSIZE == 16
        uint4 v = make_uint4(i, i+1, i+2, u2);
        *((uint4*)(base + off)) = v;
#elif WSIZE == 32
        uint4 v0 = make_uint4(i, i+1, i+2, u2);
        uint4 v1 = make_uint4(i+3, i+4, i+5, i+u2);
        *((uint4*)(base + off)) = v0;
        *((uint4*)(base + off + 16)) = v1;
#elif WSIZE == 128
        uint4 v = make_uint4(i, i+1, i+2, u2);
        *((uint4*)(base + off + 0)) = v;
        *((uint4*)(base + off + 16)) = v;
        *((uint4*)(base + off + 32)) = v;
        *((uint4*)(base + off + 48)) = v;
        *((uint4*)(base + off + 64)) = v;
        *((uint4*)(base + off + 80)) = v;
        *((uint4*)(base + off + 96)) = v;
        *((uint4*)(base + off + 112)) = v;
#endif
    }
}
