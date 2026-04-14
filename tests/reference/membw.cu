// Memory bandwidth benchmark kernels
// arg0 = number of DWORDs to process
// arg1 = mode: 0=read-only, 1=write-only (memset), 2=copy (read+write)
// arg2 = nonzero to validate (use with -i for init, and -T 0)
//
// Change these to float2/int2/etc as needed:
#ifndef VEC_T
#define VEC_T float4
#define VEC_DWORDS 4
#endif

// Define USE_RESTRICT for __restrict__ on all pointers (may help compiler optimization)
#ifdef USE_RESTRICT
#define R __restrict__
#else
#define R
#endif

// init: set up known data for validation. Invoke with -i flag.
extern "C" __global__ void init(float * R A, float * R B, float * R C, int num_dwords, int mode, int arg2) {
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int *Au = reinterpret_cast<unsigned int*>(A);
    unsigned int *Cu = reinterpret_cast<unsigned int*>(C);

    // Fill A with known pattern, C with sentinel
    for (int i = tid; i < num_dwords; i += stride) {
        Au[i] = i + 1;         // Source: known non-zero incrementing pattern
        Cu[i] = 0xCAFEBABEu;   // Sentinel: verifies read-only doesn't corrupt output
    }

    // Zero validation counters in B: [0]=errors, [1]=blocks_done, [2..4]=first error info
    if (tid < 8) {
        unsigned int *Bu = reinterpret_cast<unsigned int*>(B);
        Bu[tid] = (tid == 2) ? 0xFFFFFFFFu : 0u;  // B[2] = "no error" sentinel
    }
}

extern "C" __global__ void kernel(float * R A, float * R B, float * R C, int num_dwords, int mode, int arg2) {
    int num_vecs = num_dwords / VEC_DWORDS;
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    VEC_T *src = reinterpret_cast<VEC_T*>(A);
    VEC_T *dst = reinterpret_cast<VEC_T*>(C);

    // === Main operation ===
    if (mode == 0) {
        // Read-only: accumulate ALL components to force full vector loads
        // (using only one component lets the compiler emit scalar loads!)
        float accum = 0.0f;
        for (int i = tid; i < num_vecs; i += stride) {
            VEC_T v = src[i];
            float *vf = reinterpret_cast<float*>(&v);
            for (int d = 0; d < VEC_DWORDS; d++) accum += vf[d];
        }
        if (__float_as_int(accum) == 0xDEADBEEF)
            dst[tid] = *reinterpret_cast<VEC_T*>(&accum);
    } else if (mode == 1) {
        // Write-only (memset): fill C with 1.0f
        VEC_T val;
        float *vf = reinterpret_cast<float*>(&val);
        for (int d = 0; d < VEC_DWORDS; d++) vf[d] = 1.0f;
        for (int i = tid; i < num_vecs; i += stride) {
            dst[i] = val;
        }
    } else {
        // Copy: A -> C
        for (int i = tid; i < num_vecs; i += stride) {
            dst[i] = src[i];
        }
    }

    // === Validation (only when arg2 != 0, use with -i and -T 0) ===
    if (!arg2) return;

    __syncthreads();

    unsigned int *Bu = reinterpret_cast<unsigned int*>(B);
    __shared__ int block_errors;
    if (threadIdx.x == 0) block_errors = 0;
    __syncthreads();

    int my_errors = 0;
    for (int i = tid; i < num_vecs && my_errors < 8; i += stride) {
        VEC_T actual_vec = dst[i];
        float *af = reinterpret_cast<float*>(&actual_vec);

        for (int d = 0; d < VEC_DWORDS && my_errors < 8; d++) {
            unsigned int actual_u = __float_as_uint(af[d]);
            unsigned int expected_u;
            int dword_idx = i * VEC_DWORDS + d;

            if (mode == 0)      expected_u = 0xCAFEBABEu;                      // Sentinel from init
            else if (mode == 1) expected_u = 0x3F800000u;                       // 1.0f
            else                expected_u = static_cast<unsigned int>(dword_idx + 1); // Init pattern

            if (actual_u != expected_u) {
                my_errors++;
                // First error globally: try to claim B[2] via atomicCAS
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

        // Last block prints summary
        if (completed == gridDim.x) {
            unsigned int total = *reinterpret_cast<volatile unsigned int*>(Bu);
            const char *name = (mode == 0) ? "read-only" : (mode == 1) ? "memset" : "memcpy";
            if (total == 0) {
                printf("Validation PASS (%s, %d DWORDs)\n", name, num_dwords);
            } else {
                unsigned int ei = *reinterpret_cast<volatile unsigned int*>(Bu + 2);
                unsigned int ee = *reinterpret_cast<volatile unsigned int*>(Bu + 3);
                unsigned int ea = *reinterpret_cast<volatile unsigned int*>(Bu + 4);
                printf("Validation FAIL (%s): %u errors (first @ DWORD[%u]: expected 0x%08X got 0x%08X)\n",
                       name, total, ei, ee, ea);
            }
        }
    }
}
