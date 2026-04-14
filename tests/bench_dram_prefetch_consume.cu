// DRAM → SM LOAD benchmark, "prefetch ADVANCE_BYTES ahead then consume" style.
//
// Design:
//   - Consumer: every thread does ld.global.v8.f32 at its own tile (32 B/thread).
//     Across a warp: 32 threads × 32 B = 1 KiB consumed per warp per iter.
//   - Prefetcher: only 1-in-PF_STRIDE_LANES threads issues a scalar
//     ld.global.L2::256B.f32 at (their tile addr + ADVANCE_BYTES). Each such
//     load triggers a 256-byte HBM → L2 sector fill. Lanes 0, PF_STRIDE, 2*PF_STRIDE,
//     ... cover PF_STRIDE × 32 B = 256 B of data per prefetch when PF_STRIDE=8.
//
// Balance: PF_STRIDE_LANES=8 → 4 prefetch instructions per warp (lanes 0,8,16,24)
// to cover all 1 KiB that the 32-thread consume later reads. Each prefetch drives
// 256 B of HBM traffic; 4 prefetches × 256 B = 1 KiB HBM per warp = consume rate.
//
// ADVANCE_BYTES should be larger than HBM round-trip × bandwidth (so the L2 fill
// lands before consumption reaches that address) and smaller than L2 capacity
// (so it hasn't been evicted). B300 L2 ≈ 60 MiB/die; sweet spot expected 8-40 MiB.

#ifndef ADVANCE_BYTES
#define ADVANCE_BYTES (32 * 1024 * 1024)
#endif
#ifndef PF_STRIDE_LANES
// Must evenly divide 32. Each prefetch covers PF_STRIDE_LANES × 32 B of future
// consume. PF_STRIDE_LANES = 8 → 4 prefetches per warp, 256 B covered each.
#define PF_STRIDE_LANES 8
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef UNROLL
#define UNROLL 1
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(const float* __restrict__ A,
            float* __restrict__ B,
            float* __restrict__ C,
            int num_elems_v8, int seed, int unused) {

  const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const unsigned long long total_bytes = (unsigned long long)num_elems_v8 * 32ULL;
  const int lane = threadIdx.x & 31;

  float accum = 0.0f;

  #pragma unroll 1
  for (int i = tid; i < num_elems_v8; i += stride * UNROLL) {
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      int idx = i + u * stride;
      if (idx >= num_elems_v8) continue;

      // 1. Sparse prefetch — only 1-in-PF_STRIDE_LANES threads issues it
      if ((lane % PF_STRIDE_LANES) == 0) {
        unsigned long long my_byte = (unsigned long long)idx * 32ULL;
        unsigned long long pf_byte = my_byte + (unsigned long long)ADVANCE_BYTES;
        if (pf_byte < total_bytes) {
          const float* pf_addr = (const float*)((const char*)A + pf_byte);
          float dummy;
          asm volatile("ld.global.L2::256B.f32 %0, [%1];"
                       : "=f"(dummy) : "l"(pf_addr));
          accum += dummy * 0.0f;
        }
      }

      // 2. Full-rate 256-bit consume
      const float* addr = A + idx * 8;
      float r0, r1, r2, r3, r4, r5, r6, r7;
      asm("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
          : "=f"(r0),"=f"(r1),"=f"(r2),"=f"(r3),
            "=f"(r4),"=f"(r5),"=f"(r6),"=f"(r7)
          : "l"(addr));
      accum += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
    }
  }

  if (__float_as_int(accum) == seed) C[tid] = accum;
}
