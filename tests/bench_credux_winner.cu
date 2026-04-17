// Find the lowest-laneid that holds the minimum x value, then have only that
// lane execute compute+store. Investigate most-efficient SASS pattern.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    unsigned x = (unsigned)(seed ^ (tid * 0x9E3779B9u));
    unsigned acc = 0;

#if OP == 0
    // User's pattern: 2x credux_min
    unsigned mn = __reduce_min_sync(0xFFFFFFFF, x);
    bool is_min = (x == mn);
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    unsigned y = is_min ? lane : 0xFFFFFFFFu;
    unsigned winner_lane = __reduce_min_sync(0xFFFFFFFF, y);
    bool is_winner = (lane == winner_lane);
    if (is_winner) {
        // Compute + store
        for (int i = 0; i < 16; i++) acc = acc * 1664525u + 1013904223u + x;
        C[blockIdx.x] = acc;
    }

#elif OP == 1
    // Ballot + ffs: 1 credux_min + 1 ballot + 1 ffs
    unsigned mn = __reduce_min_sync(0xFFFFFFFF, x);
    bool is_min = (x == mn);
    unsigned mask = __ballot_sync(0xFFFFFFFF, is_min);
    unsigned winner_lane = __ffs(mask) - 1;  // lowest set bit
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    bool is_winner = (lane == winner_lane);
    if (is_winner) {
        for (int i = 0; i < 16; i++) acc = acc * 1664525u + 1013904223u + x;
        C[blockIdx.x] = acc;
    }

#elif OP == 2
    // Combine x and laneid into one u32 — high bits = x, low bits = laneid
    // Then a single credux_min finds (min_x, lowest_laneid) automatically
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    // Pack: (x << 5) | lane, but x is full u32 so we lose top 5 bits
    // Better: treat as 64-bit, or use top bits only
    unsigned packed = (x & 0xFFFFFFE0u) | (lane & 0x1Fu);  // lossy but ok if x has spare bits
    unsigned winner_packed = __reduce_min_sync(0xFFFFFFFF, packed);
    unsigned winner_lane = winner_packed & 0x1Fu;
    bool is_winner = (lane == winner_lane);
    if (is_winner) {
        for (int i = 0; i < 16; i++) acc = acc * 1664525u + 1013904223u + x;
        C[blockIdx.x] = acc;
    }

#elif OP == 3
    // Skip laneid SREG: use ballot+ffs to test "am I lane k" via mask test
    unsigned mn = __reduce_min_sync(0xFFFFFFFF, x);
    unsigned mask = __ballot_sync(0xFFFFFFFF, x == mn);
    // Get my lane via __activemask trick? No — just match.warp pattern
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    if (lane == __ffs(mask) - 1) {
        for (int i = 0; i < 16; i++) acc = acc * 1664525u + 1013904223u + x;
        C[blockIdx.x] = acc;
    }

#elif OP == 4
    // Even smarter: pack x in upper bits + laneid in low 5 bits, single credux_min,
    // then use __ffs of "winner" mask to identify winner without comparing lane
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    unsigned packed = (x & 0xFFFFFFE0u) | lane;
    unsigned winner = __reduce_min_sync(0xFFFFFFFF, packed);
    if (packed == winner) {
        // Each lane checks: did I win? Only 1 lane will hit this.
        for (int i = 0; i < 16; i++) acc = acc * 1664525u + 1013904223u + x;
        C[blockIdx.x] = acc;
    }

#elif OP == 5
    // Use winner_lane bits directly as the predicate, skip extra ISETP
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    unsigned packed = (x & 0xFFFFFFE0u) | lane;
    unsigned winner = __reduce_min_sync(0xFFFFFFFF, packed);
    bool is_winner = ((winner & 0x1Fu) == lane);  // compare just low 5 bits
    if (is_winner) {
        for (int i = 0; i < 16; i++) acc = acc * 1664525u + 1013904223u + x;
        C[blockIdx.x] = acc;
    }
#endif
}
