// Top-6 of 256 values: 1 warp, 32 lanes, 8 values per lane.
// Two variants: lossy-precision (value packed with lane+idx) and full-precision.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));

    // Each lane's 8 values (loaded from A indexed by lane)
    unsigned v[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        v[k] = (unsigned)(seed * (lane + 1) * (k + 1) * 0x9E3779B9u);
    }

#if OP == 0
    // ---------- LOSSY (24-bit value, 5-bit lane, 3-bit local_idx) ----------
    // Each lane's max packed = (v[k] & ~0xFF) | (lane << 3) | k for the local-best k.
    // First find each lane's local-max packed value:
    unsigned local_max = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        unsigned packed = (v[k] & 0xFFFFFF00u) | (lane << 3) | k;
        if (packed > local_max) local_max = packed;
    }

    // Now extract top-6 by repeated CREDUX.MAX + mask:
    unsigned top[6];
    unsigned cur = local_max;
    #pragma unroll
    for (int n = 0; n < 6; n++) {
        // Find the warp-wide max
        unsigned winner = __reduce_max_sync(0xFFFFFFFF, cur);
        top[n] = winner;
        // Lane that owned 'winner' replaces its value with its NEXT-best local value
        if (cur == winner) {
            // Re-scan for next-best (excluding the one that won)
            unsigned won_idx = winner & 7u;
            unsigned next = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                if (k != (int)won_idx) {
                    unsigned packed = (v[k] & 0xFFFFFF00u) | (lane << 3) | k;
                    if (packed > next) next = packed;
                }
            }
            cur = next;
            // Mark used local_idx by zeroing v[won_idx]
            v[won_idx & 7] = 0;
        }
    }

    // Lane 0 writes the 6 winners
    if (lane == 0) {
        #pragma unroll
        for (int n = 0; n < 6; n++) C[blockIdx.x * 6 + n] = top[n];
    }

#elif OP == 1
    // ---------- FULL PRECISION (32-bit value + separate metadata) ----------
    // Use 2 CREDUX per iter: one for value, one to find winner lane+idx
    // Each lane has 8 (value, idx) tuples; track local-best with full precision.
    unsigned local_max = 0;
    unsigned local_idx = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        if (v[k] > local_max) { local_max = v[k]; local_idx = k; }
    }

    unsigned top_v[6];
    unsigned top_meta[6];  // (lane << 3) | local_idx
    unsigned cur_v = local_max;
    unsigned cur_meta = (lane << 3) | local_idx;

    #pragma unroll
    for (int n = 0; n < 6; n++) {
        unsigned global_max = __reduce_max_sync(0xFFFFFFFF, cur_v);
        // Find winner lane: pack (one_if_owns_max, lane) and CREDUX.MIN
        // (lane with max wins; among ties, lowest lane wins)
        unsigned packed = (cur_v == global_max) ? (lane) : 0xFFFFFFFFu;
        unsigned winner_lane = __reduce_min_sync(0xFFFFFFFF, packed);
        top_v[n] = global_max;
        // Get the meta from winner via shfl
        top_meta[n] = __shfl_sync(0xFFFFFFFF, cur_meta, winner_lane);
        // Winner finds its next-best value
        if (lane == winner_lane) {
            unsigned won_k = cur_meta & 7u;
            v[won_k] = 0;  // mark used
            unsigned next_v = 0;
            unsigned next_k = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                if (v[k] > next_v) { next_v = v[k]; next_k = k; }
            }
            cur_v = next_v;
            cur_meta = (lane << 3) | next_k;
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int n = 0; n < 6; n++) {
            C[blockIdx.x * 12 + n*2] = top_v[n];
            C[blockIdx.x * 12 + n*2 + 1] = top_meta[n];
        }
    }

#elif OP == 3
    // ---------- LOSSY no pre-sort: find local-max each iter, mask after win ----------
    unsigned top[6];
    #pragma unroll
    for (int n = 0; n < 6; n++) {
        // Find current local max with packed metadata
        unsigned local_max = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned packed = (v[k] & 0xFFFFFF00u) | (lane << 3) | k;
            if (packed > local_max) local_max = packed;
        }
        unsigned winner = __reduce_max_sync(0xFFFFFFFF, local_max);
        top[n] = winner;
        if (local_max == winner) {
            // Find which k was the winner and mask it
            unsigned won_k = winner & 7u;
            v[won_k] = 0;
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int n = 0; n < 6; n++) C[blockIdx.x * 6 + n] = top[n];
    }

#elif OP == 2
    // ---------- LOSSY but cleaner: maintain "next-best" as we extract ----------
    // First sort each lane's 8 values (in registers) to get local top-6 ordered
    // (insertion sort 8 vals fits in a few inst). Then extract by repeated CREDUX.MAX.
    // But sorting 8 vals takes ~28 compares. Instead: just track 6 best per lane.

    // Pre-sort top-6 within each lane (insertion sort)
    unsigned local_top[6] = {0,0,0,0,0,0};
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        unsigned vk = v[k];
        // Insert into local_top sorted descending
        #pragma unroll
        for (int s = 0; s < 6; s++) {
            if (vk > local_top[s]) {
                unsigned tmp = local_top[s];
                local_top[s] = vk;
                vk = tmp;
            }
        }
    }

    // Extract top-6 across warp by 6 CREDUX.MAX iterations
    // Each lane's "current candidate" cycles through its sorted local_top
    unsigned next_idx = 0;
    unsigned cur = local_top[0];

    unsigned top[6];
    #pragma unroll
    for (int n = 0; n < 6; n++) {
        // Pack: (value & ~0x1F) | lane — but lane already unique per "row"
        unsigned packed = (cur & 0xFFFFFFE0u) | lane;
        unsigned winner = __reduce_max_sync(0xFFFFFFFF, packed);
        top[n] = winner;
        if (packed == winner) {
            // Advance to next local-best
            next_idx++;
            cur = (next_idx < 6) ? local_top[next_idx] : 0;
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int n = 0; n < 6; n++) C[blockIdx.x * 6 + n] = top[n];
    }
#endif
}
