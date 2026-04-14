// Read-only bandwidth test using scalar 32-bit loads with L2 prefetch hints
// Each thread loads 4 bytes but the prefetch hint pulls PREFETCH_BYTES into L2.
// Threads are spaced PREFETCH_BYTES apart so prefetch regions tile contiguously.
//
// arg0 = number of DWORDs to process
//
// Compile-time defines:
//   UNROLL: manual unroll factor (default 1)
//   BLOCK_SIZE: threads per block (default 1024)
//   MIN_BLOCKS: min blocks per SM for __launch_bounds__ (default 1)
//   PREFETCH_BYTES: 128 or 256 (default 256)

#ifndef UNROLL
#define UNROLL 1
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef PREFETCH_BYTES
#define PREFETCH_BYTES 256
#endif

#define PREFETCH_FLOATS (PREFETCH_BYTES / 4)

#define _STR(x) #x
#define STR(x) _STR(x)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS) void kernel(
    const float * __restrict__ A, float *B, float *C,
    int num_dwords, int arg1, int arg2)
{
    int num_chunks = num_dwords / PREFETCH_FLOATS;
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float accum = 0.0f;
    int i = tid;

    // Main unrolled loop
    #pragma unroll 1
    for (; i + (UNROLL - 1) * stride < num_chunks; i += stride * UNROLL) {
        float vals[UNROLL];
        #pragma unroll UNROLL
        for (int u = 0; u < UNROLL; u++) {
            const float *addr = A + (i + u * stride) * PREFETCH_FLOATS;
            asm(
                "ld.global.L2::" STR(PREFETCH_BYTES) "B.f32 %0, [%1];"
                : "=f"(vals[u])
                : "l"(addr)
            );
        }
        #pragma unroll UNROLL
        for (int u = 0; u < UNROLL; u++)
            accum += vals[u];
#ifdef SYNC
        __syncthreads();
#endif
    }

    // Remainder
    #pragma unroll 1
    for (; i < num_chunks; i += stride) {
        float val;
        const float *addr = A + i * PREFETCH_FLOATS;
        asm(
            "ld.global.L2::" STR(PREFETCH_BYTES) "B.f32 %0, [%1];"
            : "=f"(val)
            : "l"(addr)
        );
        accum += val;
    }

    // Prevent dead code elimination
    if (__float_as_int(accum) == 0xDEADBEEF) {
        C[tid] = accum;
    }
}
