#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned acc = 0;
    #pragma unroll 1
    for (int it = 0; it < 1024; it++) {
        unsigned v[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            v[k] = (unsigned)(seed + it * 31 + lane * 7 + k * 13);
        }

#if OP == 2
        // Lossy + presorted
        unsigned local_top[6] = {0,0,0,0,0,0};
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned vk = v[k];
            #pragma unroll
            for (int s = 0; s < 6; s++) {
                if (vk > local_top[s]) { unsigned t=local_top[s]; local_top[s]=vk; vk=t; }
            }
        }
        unsigned next_idx = 0, cur = local_top[0];
        #pragma unroll
        for (int n = 0; n < 6; n++) {
            unsigned packed = (cur & 0xFFFFFFE0u) | lane;
            unsigned winner = __reduce_max_sync(0xFFFFFFFF, packed);
            acc ^= winner;
            if (packed == winner) {
                next_idx++;
                cur = (next_idx < 6) ? local_top[next_idx] : 0;
            }
        }
#elif OP == 4
        // Full precision: 2× CREDUX no SHFL
        unsigned local_max = 0, local_idx = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) if (v[k] > local_max) { local_max = v[k]; local_idx = k; }
        unsigned cur_v = local_max;
        unsigned cur_meta = (lane << 3) | local_idx;
        #pragma unroll
        for (int n = 0; n < 6; n++) {
            unsigned gmax = __reduce_max_sync(0xFFFFFFFF, cur_v);
            unsigned my_meta_or = (cur_v == gmax) ? cur_meta : 0xFFFFFFFFu;
            unsigned wmeta = __reduce_min_sync(0xFFFFFFFF, my_meta_or);
            acc ^= gmax ^ wmeta;
            if (cur_meta == wmeta) {
                unsigned won_k = wmeta & 7u;
                v[won_k] = 0;
                unsigned next_v = 0, next_k = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) if (v[k] > next_v) { next_v = v[k]; next_k = k; }
                cur_v = next_v;
                cur_meta = (lane << 3) | next_k;
            }
        }
#elif OP == 5
        // Full precision: ballot+ffs+shfl
        unsigned local_max = 0, local_idx = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) if (v[k] > local_max) { local_max = v[k]; local_idx = k; }
        unsigned cur_v = local_max;
        unsigned cur_meta = (lane << 3) | local_idx;
        #pragma unroll
        for (int n = 0; n < 6; n++) {
            unsigned gmax = __reduce_max_sync(0xFFFFFFFF, cur_v);
            unsigned mask = __ballot_sync(0xFFFFFFFF, cur_v == gmax);
            unsigned wlane = __ffs(mask) - 1;
            unsigned wmeta = __shfl_sync(0xFFFFFFFF, cur_meta, wlane);
            acc ^= gmax ^ wmeta;
            if (lane == wlane) {
                unsigned won_k = cur_meta & 7u;
                v[won_k] = 0;
                unsigned next_v = 0, next_k = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) if (v[k] > next_v) { next_v = v[k]; next_k = k; }
                cur_v = next_v;
                cur_meta = (lane << 3) | next_k;
            }
        }
#endif
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (lane == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = acc;
    }
}
