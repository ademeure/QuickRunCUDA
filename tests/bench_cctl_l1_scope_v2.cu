// V2: investigate the bimodal cross-CTA CCTL effect.
// Variations:
//   N_CCTL — how many CCTLs the invalidator issues in tight sequence
//   POST_DELAY — invalidator nanosleeps AFTER CCTL before signaling completion
//   PRE_DELAY  — warmer nanosleeps AFTER receiving completion, BEFORE re-reading

#ifndef N_LINES
#define N_LINES 32
#endif
#ifndef N_OUTER
#define N_OUTER 50
#endif
#ifndef N_CCTL
#define N_CCTL 1
#endif
#ifndef POST_DELAY
#define POST_DELAY 0
#endif
#ifndef PRE_DELAY
#define PRE_DELAY 0
#endif

extern "C" __global__ __launch_bounds__(64, 8)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    int tid = threadIdx.x, warp = tid / 32, lane = tid & 31, bid = blockIdx.x;
    int* workspace = (int*)A;
    unsigned int* sm_pair = (unsigned int*)(workspace + 67108864 - 2048);
    volatile unsigned int* flags = (volatile unsigned int*)(workspace + 67108864 - 1024);

    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    int role = -1;
    if (warp == 0 && lane == 0) {
        unsigned int prev = atomicCAS(&sm_pair[smid * 4], 0u, (unsigned)bid + 1);
        if (prev == 0u) role = 0;
        else {
            unsigned int prev2 = atomicCAS(&sm_pair[smid * 4 + 1], 0u, (unsigned)bid + 1);
            if (prev2 == 0u) role = 1;
        }
    }
    role = __shfl_sync(0xffffffff, role, 0);
    if (role == -1) return;
    if (warp != 0) return;

    int* my_lines = workspace + (smid * 0x10000 / 4);
    long long total_dt = 0;
    int v = lane + 1;
    volatile unsigned int* my_flag = &flags[smid * 8];
    if (lane == 0 && role == 0) my_flag[0] = 0;
    __syncwarp();

    for (int it = 0; it < N_OUTER; it++) {
        if (role == 0) {
            // Fill L1
            int s = 0;
            #pragma unroll
            for (int k = 0; k < N_LINES; k++) {
                int x;
                asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(my_lines + k * 32 + lane));
                s ^= x;
            }
            if (s == 0xDEADBEEF) C[bid] = (float)s;

            // Signal invalidator
            if (lane == 0) {
                __threadfence_block();
                my_flag[0] = it + 1;
                while (my_flag[1] != it + 1) {}
            }
            __syncwarp();

#if PRE_DELAY > 0
            asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)PRE_DELAY));
#endif

            // Re-read same lines
            unsigned long long t0, t1;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
            int s2 = 0;
            #pragma unroll
            for (int k = 0; k < N_LINES; k++) {
                int x;
                asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(my_lines + k * 32 + lane));
                s2 ^= x;
            }
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
            v ^= s2;
            total_dt += (long long)(t1 - t0);
        } else {
            if (lane == 0) {
                while (my_flag[0] != it + 1) {}
                #pragma unroll
                for (int k = 0; k < N_CCTL; k++) {
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                }
#if POST_DELAY > 0
                asm volatile("nanosleep.u32 %0;" :: "r"((unsigned)POST_DELAY));
#endif
                __threadfence_block();
                my_flag[1] = it + 1;
            }
            __syncwarp();
        }
    }

    if (role == 0 && lane == 0) {
        ((unsigned long long*)C)[1024 + smid] = (unsigned long long)total_dt;
    }
    if (v == 0xCAFEBABE) C[1] = (float)v;
}
