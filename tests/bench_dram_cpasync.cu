// Copy via cp.async through shared memory, double-buffered.
// Each block stages an input tile into smem using cp.async, then writes it
// back to DRAM. Idea: separate the load issue from the store issue.
//
// THREADS must be defined via -H. ELEMS_PER_BLOCK = THREADS * UNROLL vec16.
//
// UNROLL:  int4s per thread per stage
// STAGES:  double-buffer stages (2 or 3 typical)

#ifndef UNROLL
#define UNROLL 4
#endif
#ifndef STAGES
#define STAGES 2
#endif

extern "C" __global__ void kernel(const float* __restrict__ A_,
                                  float* __restrict__ B_,
                                  float* __restrict__ C_,
                                  int num_elems, int unused_1, int unused_2) {
  // vec16 (int4-like) lanes
  const int4* A = (const int4*)A_;
  int4*       C = (int4*)C_;

  const int tid   = threadIdx.x;
  const int bs    = blockDim.x;
  const int bdim  = gridDim.x;
  const int elems_per_block = bs * UNROLL;

  extern __shared__ __align__(16) int4 smem[]; // size = STAGES * elems_per_block * 16B

  int base = blockIdx.x * elems_per_block;
  int total = num_elems;

  // Persistent grid-stride. Process one tile per iter.
  const int tile_stride = bdim * elems_per_block;

  // Prefetch first STAGES-1 tiles
  int prefetch_tile = base;
  for (int s = 0; s < STAGES - 1 && prefetch_tile < total; s++, prefetch_tile += tile_stride) {
    #pragma unroll
    for (int k = 0; k < UNROLL; k++) {
      int idx = prefetch_tile + tid + k * bs;
      int4* smem_dst = &smem[s * elems_per_block + tid + k * bs];
      if (idx < total) {
        asm volatile(
          "cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
          : : "l"((unsigned long long)__cvta_generic_to_shared(smem_dst)),
              "l"((unsigned long long)(A + idx))
        );
      }
    }
  }
  asm volatile("cp.async.commit_group;\n" ::);

  for (int tile = base; tile < total; tile += tile_stride) {
    // Prefetch next tile (if any)
    if (prefetch_tile < total) {
      int stage = (tile / tile_stride + STAGES - 1) % STAGES;
      #pragma unroll
      for (int k = 0; k < UNROLL; k++) {
        int idx = prefetch_tile + tid + k * bs;
        int4* smem_dst = &smem[stage * elems_per_block + tid + k * bs];
        if (idx < total) {
          asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
            : : "l"((unsigned long long)__cvta_generic_to_shared(smem_dst)),
                "l"((unsigned long long)(A + idx))
          );
        }
      }
      prefetch_tile += tile_stride;
    }
    asm volatile("cp.async.commit_group;\n" ::);

    // Wait for the oldest outstanding group
    asm volatile("cp.async.wait_group %0;\n" : : "n"(STAGES - 1));
    __syncthreads();

    // Store back from the current stage
    int stage = (tile / tile_stride) % STAGES;
    #pragma unroll
    for (int k = 0; k < UNROLL; k++) {
      int idx = tile + tid + k * bs;
      if (idx < total) {
        int4 v = smem[stage * elems_per_block + tid + k * bs];
        asm volatile(
          "st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
          : : "l"((unsigned long long)(C + idx)),
              "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        );
      }
    }
    __syncthreads();
  }
}
