// L2/DRAM data-dependent power test
// Each warp does .v8 1024B loads from buffer
// Buffer is pre-initialized with controlled pattern
// Args: buffer_size_bytes (in -A buffer arg), iters (-0)
// Pattern is set up by INIT kernel based on -1 mode
// Cache hint .cg bypasses L1 (forces L2 hit if warm, DRAM if cold)

#ifndef PATTERN_MODE
#define PATTERN_MODE 0
#endif

// PATTERN_MODE values:
//  0: all same byte (0x12121212)
//  1: bytes alternating 0x00 / 0xFF (0xFF00FF00)
//  2: random per word
//  3: 32B chunks unique, repeated every 32B
//  4: 128B chunks unique, repeated every 128B
//  5: 1024B chunks unique, repeated every 1024B
//  6: all zero
//  7: all FF (0xFFFFFFFF)
//  8: 4B chunks unique, repeated every 4B (every word same)
//  9: alternating words 0x00 / 0xFFFFFFFF

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int u1, int n_words) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned* p = (unsigned*)A;
    for (int i = idx; i < n_words; i += stride) {
        unsigned v;
        switch (PATTERN_MODE) {
            case 0: v = 0x12121212u; break;
            case 1: v = 0xFF00FF00u; break;
            case 2: {
                unsigned x = i * 0x9E3779B1u + 0xCAFEBABEu;
                x ^= x >> 16; x *= 0xCAFEBABEu;
                v = x;
                break;
            }
            case 3: {
                // Within a 1024B period, the same 32B chunk is repeated.
                // 1024B = 256 words. 32B = 8 words.
                // Position within 1024B chunk: i % 256. Position within 32B: i % 8.
                int word_in_32B = i % 8;
                unsigned x = word_in_32B * 0x9E3779B1u + 0xCAFEBABEu;
                x ^= x >> 16; x *= 0xCAFEBABEu;
                v = x;
                break;
            }
            case 4: {
                int word_in_128B = i % 32;
                unsigned x = word_in_128B * 0x9E3779B1u + 0xCAFEBABEu;
                x ^= x >> 16; x *= 0xCAFEBABEu;
                v = x;
                break;
            }
            case 5: {
                int word_in_1024B = i % 256;
                unsigned x = word_in_1024B * 0x9E3779B1u + 0xCAFEBABEu;
                x ^= x >> 16; x *= 0xCAFEBABEu;
                v = x;
                break;
            }
            case 6: v = 0; break;
            case 7: v = 0xFFFFFFFFu; break;
            case 8: v = 0xCAFEBABEu; break;  // every word same value
            case 9: v = (i & 1) ? 0xFFFFFFFFu : 0u; break;  // alternating words
            case 10: { // FULL RANDOM (no repetition) - HW-quality random
                unsigned x = i;
                x = (x ^ (x >> 16)) * 0x7feb352du;
                x = (x ^ (x >> 15)) * 0x846ca68bu;
                x = x ^ (x >> 16);
                v = x;
                break;
            }
            default: v = 0;
        }
        p[i] = v;
    }
}

extern "C" __global__
void kernel(float* A, float* B, float* C, int iters, int chunks_per_warp, int unused2) {
    // .v8 loads = 32 bytes per thread, 1024B per warp
    // Multi-warp per block: warp_id = blockIdx.x * (blockDim.x/32) + (threadIdx.x / 32)
    // Each warp owns chunks_per_warp consecutive 1024B chunks
    unsigned warps_per_block = blockDim.x / 32;
    unsigned warp_in_block = threadIdx.x / 32;
    unsigned warp_id = blockIdx.x * warps_per_block + warp_in_block;
    unsigned tid = threadIdx.x % 32;  // lane within warp
    unsigned* base = (unsigned*)A;

    unsigned slice_start = warp_id * chunks_per_warp * 256;

    if (chunks_per_warp <= 0) chunks_per_warp = 1;  // safety
    unsigned acc = 0;
    for (int i = 0; i < iters; i++) {
        unsigned chunk_idx = i % chunks_per_warp;
        unsigned offset = slice_start + chunk_idx * 256 + tid * 8;
        uint4 v0, v1;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                     : "l"(&base[offset]));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v1.x), "=r"(v1.y), "=r"(v1.z), "=r"(v1.w)
                     : "l"(&base[offset + 4]));
        acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w;
    }
    if (tid == 0 && acc == 0xDEADBEEF) {
        ((unsigned*)C)[blockIdx.x] = acc;
    }
}
