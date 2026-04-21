// L2/DRAM data-dep power test v2: high-BW kernel
// 16 warps per SM × 148 SMs = 2368 warps
// Each iter does 4 chained v8 loads (4×1024B = 4KB per warp per iter, ILP=4)
// Total per iter: 2368 * 4096B = 9.5 MB read

#ifndef PATTERN_MODE
#define PATTERN_MODE 0
#endif
#ifndef ILP
#define ILP 4
#endif

extern "C" __global__ void init(float* A, float* B, float* C, int u0, int u1, int n_words) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    unsigned* p = (unsigned*)A;
    for (int i = idx; i < n_words; i += stride) {
        unsigned v;
        switch (PATTERN_MODE) {
            case 0: v = 0x12121212u; break;
            case 1: v = 0xFF00FF00u; break;
            case 2: { unsigned x = i * 0x9E3779B1u + 0xCAFEBABEu; x ^= x >> 16; x *= 0xCAFEBABEu; v = x; break; }
            case 3: { int wi = i % 8; unsigned x = wi * 0x9E3779B1u + 0xCAFEBABEu; x ^= x >> 16; x *= 0xCAFEBABEu; v = x; break; }
            case 4: { int wi = i % 32; unsigned x = wi * 0x9E3779B1u + 0xCAFEBABEu; x ^= x >> 16; x *= 0xCAFEBABEu; v = x; break; }
            case 5: { int wi = i % 256; unsigned x = wi * 0x9E3779B1u + 0xCAFEBABEu; x ^= x >> 16; x *= 0xCAFEBABEu; v = x; break; }
            case 6: v = 0; break;
            case 7: v = 0xFFFFFFFFu; break;
            case 8: v = 0xCAFEBABEu; break;
            case 9: v = (i & 1) ? 0xFFFFFFFFu : 0u; break;
            case 10: { unsigned x = i; x = (x ^ (x >> 16)) * 0x7feb352du; x = (x ^ (x >> 15)) * 0x846ca68bu; x = x ^ (x >> 16); v = x; break; }
            default: v = 0;
        }
        p[i] = v;
    }
}

extern "C" __global__ void kernel(float* A, float* B, float* C, int iters, int chunks_per_warp, int u2) {
    if (chunks_per_warp <= 0) chunks_per_warp = 1;
    unsigned warps_per_block = blockDim.x / 32;
    unsigned warp_in_block = threadIdx.x / 32;
    unsigned warp_id = blockIdx.x * warps_per_block + warp_in_block;
    unsigned tid = threadIdx.x % 32;
    unsigned* base = (unsigned*)A;

    unsigned slice_start = warp_id * chunks_per_warp * 256;
    unsigned acc = 0;

    for (int i = 0; i < iters; i++) {
        // ILP=4: do 4 v8 loads in parallel (each = 1024B per warp = 4KB total per iter)
        // Each v8 = 2 v4 loads
        uint4 a0, a1, b0, b1, c0, c1, d0, d1;
        unsigned chunk_idx_a = (i * 4 + 0) % chunks_per_warp;
        unsigned chunk_idx_b = (i * 4 + 1) % chunks_per_warp;
        unsigned chunk_idx_c = (i * 4 + 2) % chunks_per_warp;
        unsigned chunk_idx_d = (i * 4 + 3) % chunks_per_warp;
        unsigned off_a = slice_start + chunk_idx_a * 256 + tid * 8;
        unsigned off_b = slice_start + chunk_idx_b * 256 + tid * 8;
        unsigned off_c = slice_start + chunk_idx_c * 256 + tid * 8;
        unsigned off_d = slice_start + chunk_idx_d * 256 + tid * 8;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a0.x),"=r"(a0.y),"=r"(a0.z),"=r"(a0.w) : "l"(&base[off_a]));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a1.x),"=r"(a1.y),"=r"(a1.z),"=r"(a1.w) : "l"(&base[off_a+4]));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b0.x),"=r"(b0.y),"=r"(b0.z),"=r"(b0.w) : "l"(&base[off_b]));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b1.x),"=r"(b1.y),"=r"(b1.z),"=r"(b1.w) : "l"(&base[off_b+4]));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(c0.x),"=r"(c0.y),"=r"(c0.z),"=r"(c0.w) : "l"(&base[off_c]));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(c1.x),"=r"(c1.y),"=r"(c1.z),"=r"(c1.w) : "l"(&base[off_c+4]));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d0.x),"=r"(d0.y),"=r"(d0.z),"=r"(d0.w) : "l"(&base[off_d]));
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(d1.x),"=r"(d1.y),"=r"(d1.z),"=r"(d1.w) : "l"(&base[off_d+4]));
        acc ^= (a0.x ^ a0.y ^ a0.z ^ a0.w ^ a1.x ^ a1.y ^ a1.z ^ a1.w
              ^ b0.x ^ b0.y ^ b0.z ^ b0.w ^ b1.x ^ b1.y ^ b1.z ^ b1.w
              ^ c0.x ^ c0.y ^ c0.z ^ c0.w ^ c1.x ^ c1.y ^ c1.z ^ c1.w
              ^ d0.x ^ d0.y ^ d0.z ^ d0.w ^ d1.x ^ d1.y ^ d1.z ^ d1.w);
    }
    if (tid == 0 && acc == 0xDEADBEEF) ((unsigned*)C)[blockIdx.x] = acc;
}
