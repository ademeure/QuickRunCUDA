// Memory bandwidth benchmark kernels using explicit PTX load/store
// arg0 = number of DWORDs to process
//
// Compile-time defines:
//   MODE: 0=read-only, 1=write-only (memset), 2=copy (read+write)
//   PTX_BITS: 128 or 256
//   UNROLL: manual unroll factor (default 1 = no unrolling)
//   PTX_CACHE: optional cache hint string, e.g. .L2::evict_last
//   USE_RESTRICT: define for __restrict__ on all pointers
//   BLOCK_SIZE: threads per block for __launch_bounds__ (default 1024)
//   VALIDATE: define to enable validation (use with -i for init, and -T 0)
#ifndef MODE
#define MODE 0
#endif
#ifndef PTX_BITS
#define PTX_BITS 128
#endif
#ifndef PTX_CACHE
#define PTX_CACHE
#endif
#ifndef UNROLL
#define UNROLL 1
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef CLUSTER_SIZE
#define CLUSTER_SIZE 1
#endif
#if CLUSTER_SIZE > 1
#define CLUSTER_ATTR __cluster_dims__(CLUSTER_SIZE, 1, 1)
#else
#define CLUSTER_ATTR
#endif

#ifdef USE_RESTRICT
#define R __restrict__
#else
#define R
#endif

#define PTX_DWORDS (PTX_BITS / 32)
#define _STR(x) #x
#define STR(x) _STR(x)

static constexpr int mode = MODE;
static constexpr const char *mode_name = (MODE == 0) ? "read-only" : (MODE == 1) ? "memset" : "memcpy";

__device__ __forceinline__ void ptx_load(float *dst, const float *addr) {
#if PTX_BITS == 256
    asm(
        "ld.global" STR(PTX_CACHE) ".v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "l"(addr)
    );
#else
    asm(
        "ld.global" STR(PTX_CACHE) ".v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "l"(addr)
    );
#endif
}

__device__ __forceinline__ void ptx_store(float *addr, const float *src) {
#if PTX_BITS == 256
    asm volatile(
        "st.global" STR(PTX_CACHE) ".v8.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "l"(addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]),
           "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7])
    );
#else
    asm volatile(
        "st.global" STR(PTX_CACHE) ".v4.f32 [%0], {%1, %2, %3, %4};"
        :: "l"(addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3])
    );
#endif
}

#ifdef VALIDATE
extern "C" __global__ void init(float * R A, float * R B, float * R C, int num_dwords, int arg1, int arg2) {
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int *Au = reinterpret_cast<unsigned int*>(A);
    unsigned int *Cu = reinterpret_cast<unsigned int*>(C);

    for (int i = tid; i < num_dwords; i += stride) {
        Au[i] = i + 1;
        Cu[i] = 0xCAFEBABEu;
    }

    if (tid < 8) {
        unsigned int *Bu = reinterpret_cast<unsigned int*>(B);
        Bu[tid] = (tid == 2) ? 0xFFFFFFFFu : 0u;
    }
}
#endif

extern "C" __global__ CLUSTER_ATTR __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS) void kernel(float * R A, float * R B, float * R C, int num_dwords, int arg1, int arg2) {
    int num_vecs = num_dwords / PTX_DWORDS;

#ifdef LOCAL_STRIDE
    // Block-local striding: each block owns a contiguous chunk, stride = blockDim.x
    // Unrolled accesses are blockDim.x apart (~32KB for t1024 PTX256) instead of
    // gridDim.x*blockDim.x apart (~4.8MB for 148 blocks)
    int stride = blockDim.x;
    int block_chunk = (num_vecs / gridDim.x) & ~(blockDim.x * UNROLL - 1); // align to unroll
    int block_start = blockIdx.x * block_chunk;
    int block_size = (blockIdx.x == gridDim.x - 1) ? (num_vecs - block_start) : block_chunk;
    int block_end = block_size;  // loop limit is relative to block_start
    int i = threadIdx.x;
    #define VEC_IDX(idx) (block_start + (idx))
#else
    // Grid-stride: all blocks interleave, stride = gridDim.x * blockDim.x
    int stride = gridDim.x * blockDim.x;
    int block_end = num_vecs;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    #define VEC_IDX(idx) (idx)
#endif

    if constexpr (mode == 0) {
        // Read-only: load N vectors, accumulate, then next batch
        float accum = 0.0f;
        float buf[UNROLL][PTX_DWORDS];
        // Main unrolled loop
        #pragma unroll 1
        for (; i + (UNROLL - 1) * stride < block_end; i += stride * UNROLL) {
            #pragma unroll UNROLL
            for (int u = 0; u < UNROLL; u++)
                ptx_load(buf[u], A + VEC_IDX(i + u * stride) * PTX_DWORDS);
            #pragma unroll UNROLL
            for (int u = 0; u < UNROLL; u++)
                for (int d = 0; d < PTX_DWORDS; d++)
                    accum += buf[u][d];
#ifdef SYNC
            __syncthreads();
#endif
        }
        // Remainder
        #pragma unroll 1
        for (; i < block_end; i += stride) {
            ptx_load(buf[0], A + VEC_IDX(i) * PTX_DWORDS);
            for (int d = 0; d < PTX_DWORDS; d++)
                accum += buf[0][d];
        }
        if (__float_as_int(accum) == 0xDEADBEEF) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            ptx_store(C + tid * PTX_DWORDS, buf[0]);
        }
    } else if constexpr (mode == 1) {
        // Write-only (memset): store N vectors per iteration
        float val[PTX_DWORDS];
        for (int d = 0; d < PTX_DWORDS; d++) val[d] = 1.0f;
        // Main unrolled loop
        #pragma unroll 1
        for (; i + (UNROLL - 1) * stride < block_end; i += stride * UNROLL) {
            #pragma unroll UNROLL
            for (int u = 0; u < UNROLL; u++)
                ptx_store(C + VEC_IDX(i + u * stride) * PTX_DWORDS, val);
        }
        // Remainder
        #pragma unroll 1
        for (; i < block_end; i += stride) {
            ptx_store(C + VEC_IDX(i) * PTX_DWORDS, val);
        }
    } else {
        // Copy: load N vectors, then store N vectors
        float buf[UNROLL][PTX_DWORDS];
        // Main unrolled loop
        #pragma unroll 1
        for (; i + (UNROLL - 1) * stride < block_end; i += stride * UNROLL) {
            #pragma unroll UNROLL
            for (int u = 0; u < UNROLL; u++)
                ptx_load(buf[u], A + VEC_IDX(i + u * stride) * PTX_DWORDS);
#ifdef SYNC
            __syncthreads();
#endif
            #pragma unroll UNROLL
            for (int u = 0; u < UNROLL; u++)
                ptx_store(C + VEC_IDX(i + u * stride) * PTX_DWORDS, buf[u]);
        }
        // Remainder
        #pragma unroll 1
        for (; i < block_end; i += stride) {
            ptx_load(buf[0], A + VEC_IDX(i) * PTX_DWORDS);
            ptx_store(C + VEC_IDX(i) * PTX_DWORDS, buf[0]);
        }
    }
    #undef VEC_IDX

#ifdef VALIDATE
    if (!arg2) return;

    __syncthreads();

    unsigned int *Bu = reinterpret_cast<unsigned int*>(B);
    __shared__ int block_errors;
    if (threadIdx.x == 0) block_errors = 0;
    __syncthreads();

    int my_errors = 0;
    unsigned int *Cu = reinterpret_cast<unsigned int*>(C);

    for (int i = tid; i < num_vecs && my_errors < 8; i += stride) {
        for (int d = 0; d < PTX_DWORDS && my_errors < 8; d++) {
            int dword_idx = i * PTX_DWORDS + d;
            unsigned int actual_u = Cu[dword_idx];
            unsigned int expected_u;

            if constexpr (mode == 0)      expected_u = 0xCAFEBABEu;
            else if constexpr (mode == 1) expected_u = 0x3F800000u;
            else                          expected_u = static_cast<unsigned int>(dword_idx + 1);

            if (actual_u != expected_u) {
                my_errors++;
                if (atomicCAS(Bu + 2, 0xFFFFFFFFu, static_cast<unsigned int>(dword_idx)) == 0xFFFFFFFFu) {
                    Bu[3] = expected_u;
                    Bu[4] = actual_u;
                }
            }
        }
    }

    if (my_errors > 0) atomicAdd(&block_errors, my_errors);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(Bu, static_cast<unsigned int>(block_errors));
        __threadfence();
        unsigned int completed = atomicAdd(Bu + 1, 1u) + 1;

        if (completed == gridDim.x) {
            unsigned int total = *reinterpret_cast<volatile unsigned int*>(Bu);
            if (total == 0) {
                printf("Validation PASS (%s, %d DWORDs, %d-bit PTX)\n", mode_name, num_dwords, PTX_BITS);
            } else {
                unsigned int ei = *reinterpret_cast<volatile unsigned int*>(Bu + 2);
                unsigned int ee = *reinterpret_cast<volatile unsigned int*>(Bu + 3);
                unsigned int ea = *reinterpret_cast<volatile unsigned int*>(Bu + 4);
                printf("Validation FAIL (%s, %d-bit PTX): %u errors (first @ DWORD[%u]: expected 0x%08X got 0x%08X)\n",
                       mode_name, PTX_BITS, total, ei, ee, ea);
            }
        }
    }
#endif
}
