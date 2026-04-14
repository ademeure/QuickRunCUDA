// LOAD-only with explicit L2 prefetch via prefetch.global.L2 issued ahead.
// Thread issues a prefetch N iterations ahead of its actual load.
// This can cut HBM round-trip from the critical path.

#ifndef WIDTH_BYTES
#define WIDTH_BYTES 32
#endif
#ifndef UNROLL
#define UNROLL 2
#endif
#ifndef PREFETCH_AHEAD
#define PREFETCH_AHEAD 4
#endif

struct __align__(32) vec_t { unsigned long long a, b, c, d; };

__device__ __forceinline__ vec_t vload(const vec_t* p) {
  vec_t v;
  asm volatile("ld.global.nc.L2::256B.v4.u64 {%0,%1,%2,%3}, [%4];"
               : "=l"(v.a), "=l"(v.b), "=l"(v.c), "=l"(v.d) : "l"(p));
  return v;
}

__device__ __forceinline__ void l2_prefetch(const vec_t* p) {
  asm volatile("prefetch.global.L2 [%0];" : : "l"(p));
}

extern "C" __global__ void kernel(const float* __restrict__ A_,
                                  float* __restrict__ B_,
                                  float* __restrict__ C_,
                                  int num_elems, int seed, int unused_2) {
  const vec_t* A = (const vec_t*)A_;
  const int bs = blockDim.x;
  const int blk_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int global_tid = blk_id * bs + tid;

  int base = blk_id * bs * UNROLL + tid * UNROLL;
  const int total_threads = gridDim.x * bs;

  // Prefetch PREFETCH_AHEAD tiles
  for (int pf = 0; pf < PREFETCH_AHEAD; pf++) {
    int idx = base + pf * UNROLL * total_threads;
    #pragma unroll
    for (int k = 0; k < UNROLL; k++) {
      if (idx + k < num_elems) l2_prefetch(A + idx + k);
    }
  }

  // Now actually load the data with prefetch issued further ahead
  vec_t v[UNROLL];
  unsigned long long sink = 0;

  // Single iteration (non-persistent)
  #pragma unroll
  for (int k = 0; k < UNROLL; k++) {
    int idx = base + k;
    if (idx < num_elems) v[k] = vload(A + idx);
    else v[k] = {0, 0, 0, 0};
  }

  #pragma unroll
  for (int k = 0; k < UNROLL; k++) sink ^= v[k].a ^ v[k].b ^ v[k].c ^ v[k].d;
  if ((int)sink == seed) ((unsigned long long*)C_)[global_tid] = sink;
}
