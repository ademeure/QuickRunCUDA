// Persistent kernel overhead: vs launching many tiny kernels.
// Measure cost of CTA-loop-dispatcher pattern (persistent) vs ordinary grid.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef NTILES
#define NTILES 1000
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
#if OP == 0
    // Ordinary: each block processes blockIdx tile
    unsigned tile = blockIdx.x;
    // "Work": write accumulator to C[tile]
    unsigned acc = tile ^ seed;
    for (int i = 0; i < 32; i++) acc = acc * 1664525u + 1013904223u;
    C[tile * blockDim.x + threadIdx.x] = acc;
#elif OP == 1
    // Persistent: block loops through tiles, using atomic counter
    while (true) {
        unsigned tile;
        if (threadIdx.x == 0) tile = atomicAdd(A, 1u);
        tile = __shfl_sync(0xFFFFFFFF, tile, 0);
        if (tile >= NTILES) break;
        unsigned acc = tile ^ seed;
        for (int i = 0; i < 32; i++) acc = acc * 1664525u + 1013904223u;
        C[tile * blockDim.x + threadIdx.x] = acc;
    }
#endif
}
