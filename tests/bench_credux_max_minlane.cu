// MAX of x, MIN of laneid tie-breaker.
// Trick: pack (x & ~0x1F) | (~lane & 0x1F). Then CREDUX.MAX picks highest x, lowest lane.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    unsigned x = (unsigned)(seed ^ (lane * 0x9E3779B9u));

#if OP == 0
    // Naive 2× CREDUX
    unsigned mx = __reduce_max_sync(0xFFFFFFFF, x);
    unsigned y = (x == mx) ? lane : 0xFFFFFFFFu;
    unsigned winner_lane = __reduce_min_sync(0xFFFFFFFF, y);
    if (lane == winner_lane) {
        C[blockIdx.x] = x;
    }

#elif OP == 1
    // Pack with INVERTED lane: high bits = x, low 5 bits = ~lane
    // Then CREDUX.MAX picks (max x, min lane) in one shot
    unsigned packed = (x & 0xFFFFFFE0u) | ((~lane) & 0x1Fu);
    unsigned winner = __reduce_max_sync(0xFFFFFFFF, packed);
    if (packed == winner) {
        C[blockIdx.x] = x;
    }

#elif OP == 2
    // Same as OP=1 but compare lane directly
    unsigned packed = (x & 0xFFFFFFE0u) | ((~lane) & 0x1Fu);
    unsigned winner = __reduce_max_sync(0xFFFFFFFF, packed);
    unsigned winner_lane = (~winner) & 0x1Fu;
    if (lane == winner_lane) {
        C[blockIdx.x] = x;
    }
#endif
}
