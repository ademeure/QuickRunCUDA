// DRAM bandwidth variants. Parameterized entirely via -H defines.
//
// ACCESS_MODE: 0=load-only, 1=store-only, 2=memcpy, 3=memcpy+xor
// WIDTH_BYTES: 4 | 8 | 16 | 32
// UNROLL_N:    1..32 (loads issued per iteration per thread)
// PERSISTENT:  0 (one-shot grid, consecutive-per-thread unroll)
//              1 (persistent blocks, grid-stride unroll)
// GRID_ITERS:  persistent-mode outer iteration count (computed by driver)
//
// Layout:
//   Non-persistent: thread t handles elements [t*UNROLL_N .. t*UNROLL_N+UNROLL_N-1]
//                   within its block, blocks tile the array. Unrolled loads are adjacent.
//   Persistent:     thread t loads element [global_tid + i*gsize] for i in [0,UNROLL_N).
//                   Unrolled loads stride by the full grid.

#ifndef ACCESS_MODE
#define ACCESS_MODE 2
#endif
#ifndef WIDTH_BYTES
#define WIDTH_BYTES 16
#endif
#ifndef UNROLL_N
#define UNROLL_N 4
#endif
#ifndef PERSISTENT
#define PERSISTENT 0
#endif
#ifndef GRID_ITERS
#define GRID_ITERS 1
#endif

// LD_HINT choices:
//   0 = ld.global.nc        (non-coherent, default)
//   1 = ld.global.ca        (cache-all, default for mutable loads)
//   2 = ld.global.cg        (cache-global only, skip L1)
//   3 = ld.global.cs        (cache-streaming, eviction-priority)
//   4 = ld.global.nc.L2::256B  (non-coherent + 256B L2 sector hint)
//   5 = ld.global.nc.L2::128B  (non-coherent + 128B L2 sector hint)
//   6 = ld.global.nc.L1::no_allocate  (non-coherent, skip L1)
#ifndef LD_HINT
#define LD_HINT 0
#endif

#if LD_HINT == 0
  #define LDOP_U32 "ld.global.nc.u32"
  #define LDOP_U64 "ld.global.nc.u64"
  #define LDOP_V2U64 "ld.global.nc.v2.u64"
  #define LDOP_V4U64 "ld.global.nc.v4.u64"
#elif LD_HINT == 1
  #define LDOP_U32 "ld.global.ca.u32"
  #define LDOP_U64 "ld.global.ca.u64"
  #define LDOP_V2U64 "ld.global.ca.v2.u64"
  #define LDOP_V4U64 "ld.global.ca.v4.u64"
#elif LD_HINT == 2
  #define LDOP_U32 "ld.global.cg.u32"
  #define LDOP_U64 "ld.global.cg.u64"
  #define LDOP_V2U64 "ld.global.cg.v2.u64"
  #define LDOP_V4U64 "ld.global.cg.v4.u64"
#elif LD_HINT == 3
  #define LDOP_U32 "ld.global.cs.u32"
  #define LDOP_U64 "ld.global.cs.u64"
  #define LDOP_V2U64 "ld.global.cs.v2.u64"
  #define LDOP_V4U64 "ld.global.cs.v4.u64"
#elif LD_HINT == 4
  #define LDOP_U32 "ld.global.nc.L2::256B.u32"
  #define LDOP_U64 "ld.global.nc.L2::256B.u64"
  #define LDOP_V2U64 "ld.global.nc.L2::256B.v2.u64"
  #define LDOP_V4U64 "ld.global.nc.L2::256B.v4.u64"
#elif LD_HINT == 5
  #define LDOP_U32 "ld.global.nc.L2::128B.u32"
  #define LDOP_U64 "ld.global.nc.L2::128B.u64"
  #define LDOP_V2U64 "ld.global.nc.L2::128B.v2.u64"
  #define LDOP_V4U64 "ld.global.nc.L2::128B.v4.u64"
#elif LD_HINT == 6
  #define LDOP_U32 "ld.global.nc.L1::no_allocate.u32"
  #define LDOP_U64 "ld.global.nc.L1::no_allocate.u64"
  #define LDOP_V2U64 "ld.global.nc.L1::no_allocate.v2.u64"
  #define LDOP_V4U64 "ld.global.nc.L1::no_allocate.v4.u64"
#endif

#if WIDTH_BYTES == 4
  struct __align__(4) vec_t { unsigned int a; };
  __device__ __forceinline__ vec_t vload(const vec_t* p) {
    vec_t v; asm volatile(LDOP_U32 " %0, [%1];" : "=r"(v.a) : "l"(p)); return v;
  }
  __device__ __forceinline__ void vstore(vec_t* p, vec_t v) {
    asm volatile("st.global.u32 [%0], %1;" : : "l"(p), "r"(v.a));
  }
  __device__ __forceinline__ unsigned int vsink(vec_t v) { return v.a; }
  __device__ __forceinline__ vec_t vmake(unsigned int s) { return { s }; }
  __device__ __forceinline__ vec_t vxor(vec_t v) { v.a ^= 0xAAAAAAAAu; return v; }
  #define VSINK_REG "r"
#elif WIDTH_BYTES == 8
  struct __align__(8) vec_t { unsigned long long a; };
  __device__ __forceinline__ vec_t vload(const vec_t* p) {
    vec_t v; asm volatile(LDOP_U64 " %0, [%1];" : "=l"(v.a) : "l"(p)); return v;
  }
  __device__ __forceinline__ void vstore(vec_t* p, vec_t v) {
    asm volatile("st.global.u64 [%0], %1;" : : "l"(p), "l"(v.a));
  }
  __device__ __forceinline__ unsigned long long vsink(vec_t v) { return v.a; }
  __device__ __forceinline__ vec_t vmake(unsigned long long s) { return { s }; }
  __device__ __forceinline__ vec_t vxor(vec_t v) { v.a ^= 0xAAAAAAAAAAAAAAAAull; return v; }
  #define VSINK_REG "l"
#elif WIDTH_BYTES == 16
  struct __align__(16) vec_t { unsigned long long a, b; };
  __device__ __forceinline__ vec_t vload(const vec_t* p) {
    vec_t v;
    asm volatile(LDOP_V2U64 " {%0,%1}, [%2];"
                 : "=l"(v.a), "=l"(v.b) : "l"(p));
    return v;
  }
  __device__ __forceinline__ void vstore(vec_t* p, vec_t v) {
    asm volatile("st.global.v2.u64 [%0], {%1,%2};" : : "l"(p), "l"(v.a), "l"(v.b));
  }
  __device__ __forceinline__ unsigned long long vsink(vec_t v) { return v.a ^ v.b; }
  __device__ __forceinline__ vec_t vmake(unsigned long long s) { return { s, ~s }; }
  __device__ __forceinline__ vec_t vxor(vec_t v) {
    v.a ^= 0xAAAAAAAAAAAAAAAAull; v.b ^= 0x5555555555555555ull; return v;
  }
  #define VSINK_REG "l"
#elif WIDTH_BYTES == 32
  struct __align__(32) vec_t { unsigned long long a, b, c, d; };
  __device__ __forceinline__ vec_t vload(const vec_t* p) {
    vec_t v;
    asm volatile(LDOP_V4U64 " {%0,%1,%2,%3}, [%4];"
                 : "=l"(v.a), "=l"(v.b), "=l"(v.c), "=l"(v.d) : "l"(p));
    return v;
  }
  __device__ __forceinline__ void vstore(vec_t* p, vec_t v) {
    asm volatile("st.global.v4.u64 [%0], {%1,%2,%3,%4};"
                 : : "l"(p), "l"(v.a), "l"(v.b), "l"(v.c), "l"(v.d));
  }
  __device__ __forceinline__ unsigned long long vsink(vec_t v) {
    return v.a ^ v.b ^ v.c ^ v.d;
  }
  __device__ __forceinline__ vec_t vmake(unsigned long long s) { return { s, ~s, s*7, s*13 }; }
  __device__ __forceinline__ vec_t vxor(vec_t v) {
    v.a ^= 0xAAAAAAAAAAAAAAAAull; v.b ^= 0x5555555555555555ull;
    v.c ^= 0xAAAAAAAAAAAAAAAAull; v.d ^= 0x5555555555555555ull;
    return v;
  }
  #define VSINK_REG "l"
#else
  #error "WIDTH_BYTES must be 4, 8, 16, or 32"
#endif

extern "C" __global__ void kernel(const float* __restrict__ A_,
                                  float* __restrict__ B_,
                                  float* __restrict__ C_,
                                  int num_elems, int seed, int unused_2) {
  const vec_t* A = (const vec_t*)A_;
  vec_t*       C = (vec_t*)C_;

  const int blk_size = blockDim.x;
  const int blk_id   = blockIdx.x;
  const int tid      = threadIdx.x;
  const int gsize    = blk_size * gridDim.x;
  const int global_tid = blk_id * blk_size + tid;

#if PERSISTENT
  // Grid-stride: unrolled loads stride by gsize
  decltype(vsink(vmake(0))) sink = 0;
  #pragma unroll 1
  for (int iter = 0; iter < GRID_ITERS; iter++) {
    int base = global_tid + iter * (UNROLL_N * gsize);

    vec_t v[UNROLL_N];
    #pragma unroll
    for (int k = 0; k < UNROLL_N; k++) {
      int off = base + k * gsize;
  #if ACCESS_MODE == 0
      v[k] = vload(A + off);
  #elif ACCESS_MODE == 1
      vstore(C + off, vmake((unsigned long long)(off ^ seed)));
  #elif ACCESS_MODE == 2
      v[k] = vload(A + off);
      vstore(C + off, v[k]);
  #elif ACCESS_MODE == 3
      v[k] = vload(A + off);
      vstore(C + off, vxor(v[k]));
  #endif
    }
  #if ACCESS_MODE == 0
    #pragma unroll
    for (int k = 0; k < UNROLL_N; k++) sink ^= vsink(v[k]);
  #endif
  }
  #if ACCESS_MODE == 0
    // Real conditional store: seed is a runtime kernel arg; compiler can't prove
    // sink != seed, so it must compute sink — which forces the loads.
    if ((int)sink == seed) {
      ((unsigned long long*)C)[global_tid] = sink;
    }
  #endif

#else
  // Non-persistent, consecutive-per-thread: thread t handles UNROLL_N adjacent elements
  int base = blk_id * blk_size * UNROLL_N + tid * UNROLL_N;

  vec_t v[UNROLL_N];
  #pragma unroll
  for (int k = 0; k < UNROLL_N; k++) {
    int off = base + k;
  #if ACCESS_MODE == 0
    v[k] = vload(A + off);
  #elif ACCESS_MODE == 1
    vstore(C + off, vmake((unsigned long long)(off ^ seed)));
  #elif ACCESS_MODE == 2
    v[k] = vload(A + off);
    vstore(C + off, v[k]);
  #elif ACCESS_MODE == 3
    v[k] = vload(A + off);
    vstore(C + off, vxor(v[k]));
  #endif
  }
  #if ACCESS_MODE == 0
    decltype(vsink(vmake(0))) sink = 0;
    #pragma unroll
    for (int k = 0; k < UNROLL_N; k++) sink ^= vsink(v[k]);
    if ((int)sink == seed) {
      ((unsigned long long*)C)[global_tid] = sink;
    }
  #endif
#endif
}
