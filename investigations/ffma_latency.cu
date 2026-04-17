// ffma_latency.cu — Definitive FFMA latency measurement on B300 sm_103a
//
// PURPOSE: Resolve the "4 cy vs 23 cy" FFMA latency question.
//
// METHODOLOGY:
//   - Single warp (32 threads), only thread 0 does timing
//   - All variants use asm volatile PTX to guarantee FFMA emission
//   - b and c operands loaded from __shared__ at runtime (opaque to compiler)
//   - clock64 sampled before/after the chain with #pragma unroll 1 outer loop
//   - Anti-DCE: result written via impossible-predicate guarded store
//
// SASS VERIFICATION (after compile):
//   kernel_selfop:     FFMA R0, R0, R0, RZ          (3 reads same reg — port pressure)
//   kernel_self_const: FFMA Ra, Ra, Rb, Rc           (dest=src1, src2/addend are distinct)
//   kernel_diffsrc:    FFMA Ra, Rb, Rc, Ra           (dest=addend, mul pair are distinct)
//   kernel_ilpN:       N independent FFMA Rk, Rb, Rc, Rk  (N dep chains, all different Rk)
//
// INNER/OUTER sizing:
//   Latency kernels: INNER_LAT=1024, OUTER_LAT=20 → 20480 FMAs per chain
//   ILP kernels:     INNER_ILP=16,   OUTER_ILP=1280 → same 20480 per chain
//     → 16*N FFMA insts in the unrolled body → fits in icache

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define INNER_LAT  1024
#define OUTER_LAT  20
#define INNER_ILP  16
#define OUTER_ILP  1280

// Shared shmem init helper (avoids constant-folding of b, c)
static __device__ __forceinline__ void load_bc(float &b, float &c) {
    __shared__ float sm[2];
    if (threadIdx.x == 0) { sm[0] = 1.0001f; sm[1] = 0.00001f; }
    __syncwarp();
    b = sm[0];
    c = sm[1];
}

// ---------------------------------------------------------------------------
// V1: Self-op chain: FFMA Ra, Ra, Ra, RZ   (a = a*a + 0)
//   All three float reads from same register — known to inflate 2x
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 1)
void kernel_selfop(float* R, unsigned long long* T, float iv) {
    unsigned tid = threadIdx.x;
    float a = iv + tid * 0.001f;
    unsigned long long t0, t1;

    // Warm-up (not timed)
    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_LAT; i++)
            asm volatile("fma.rn.f32 %0, %0, %0, 0f00000000;" : "+f"(a));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER_LAT; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_LAT; i++)
            asm volatile("fma.rn.f32 %0, %0, %0, 0f00000000;" : "+f"(a));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    if (__float_as_int(a) == 0x7FFFFFFF) R[tid] = a;
    if (tid == 0) T[0] = t1 - t0;
}

// ---------------------------------------------------------------------------
// V2: Self+const chain: FFMA Ra, Ra, Rb, Rc  (a = a*b + c)
//   Dep chain through Ra. Rb, Rc are distinct registers (shmem).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 1)
void kernel_self_const(float* R, unsigned long long* T, float iv) {
    unsigned tid = threadIdx.x;
    float b, c;
    load_bc(b, c);
    float a = iv + tid * 0.001f;
    unsigned long long t0, t1;

    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_LAT; i++)
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER_LAT; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_LAT; i++)
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    if (__float_as_int(a) == 0x7FFFFFFF) R[tid] = a;
    if (tid == 0) T[1] = t1 - t0;
}

// ---------------------------------------------------------------------------
// V3: Diff-sources chain: FFMA Ra, Rb, Rc, Ra  (a = b*c + a)
//   Dep chain through Ra (accumulator). Rb, Rc are distinct registers.
//   Cleanest form: Ra appears only as dest and addend, not in multiply.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 1)
void kernel_diffsrc(float* R, unsigned long long* T, float iv) {
    unsigned tid = threadIdx.x;
    float b, c;
    load_bc(b, c);
    float a = iv + tid * 0.001f;
    unsigned long long t0, t1;

    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_LAT; i++)
            asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(a) : "f"(b), "f"(c));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER_LAT; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_LAT; i++)
            asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(a) : "f"(b), "f"(c));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    if (__float_as_int(a) == 0x7FFFFFFF) R[tid] = a;
    if (tid == 0) T[2] = t1 - t0;
}

// ---------------------------------------------------------------------------
// ILP-2: two independent diff-src chains
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 1)
void kernel_ilp2(float* R, unsigned long long* T, float iv) {
    unsigned tid = threadIdx.x;
    float b, c;
    load_bc(b, c);
    float a0 = iv + tid * 0.001f;
    float a1 = a0 + 1.0f;
    unsigned long long t0, t1;

    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_ILP; i++) {
            asm volatile("fma.rn.f32 %0, %2, %3, %0;"
                         "fma.rn.f32 %1, %2, %3, %1;"
                         : "+f"(a0), "+f"(a1) : "f"(b), "f"(c));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER_ILP; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_ILP; i++) {
            asm volatile("fma.rn.f32 %0, %2, %3, %0;"
                         "fma.rn.f32 %1, %2, %3, %1;"
                         : "+f"(a0), "+f"(a1) : "f"(b), "f"(c));
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    float s = a0 + a1;
    if (__float_as_int(s) == 0x7FFFFFFF) R[tid] = s;
    if (tid == 0) T[3] = t1 - t0;
}

// ---------------------------------------------------------------------------
// ILP-4: four independent diff-src chains
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 1)
void kernel_ilp4(float* R, unsigned long long* T, float iv) {
    unsigned tid = threadIdx.x;
    float b, c;
    load_bc(b, c);
    float a0 = iv + tid * 0.001f;
    float a1 = a0 + 1.0f, a2 = a0 + 2.0f, a3 = a0 + 3.0f;
    unsigned long long t0, t1;

    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_ILP; i++) {
            asm volatile("fma.rn.f32 %0, %4, %5, %0;"
                         "fma.rn.f32 %1, %4, %5, %1;"
                         "fma.rn.f32 %2, %4, %5, %2;"
                         "fma.rn.f32 %3, %4, %5, %3;"
                         : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3) : "f"(b), "f"(c));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER_ILP; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_ILP; i++) {
            asm volatile("fma.rn.f32 %0, %4, %5, %0;"
                         "fma.rn.f32 %1, %4, %5, %1;"
                         "fma.rn.f32 %2, %4, %5, %2;"
                         "fma.rn.f32 %3, %4, %5, %3;"
                         : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3) : "f"(b), "f"(c));
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    float s = a0 + a1 + a2 + a3;
    if (__float_as_int(s) == 0x7FFFFFFF) R[tid] = s;
    if (tid == 0) T[4] = t1 - t0;
}

// ---------------------------------------------------------------------------
// ILP-8: eight independent diff-src chains
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 1)
void kernel_ilp8(float* R, unsigned long long* T, float iv) {
    unsigned tid = threadIdx.x;
    float b, c;
    load_bc(b, c);
    float a0 = iv + tid * 0.001f;
    float a1 = a0 + 1.0f, a2 = a0 + 2.0f, a3 = a0 + 3.0f;
    float a4 = a0 + 4.0f, a5 = a0 + 5.0f, a6 = a0 + 6.0f, a7 = a0 + 7.0f;
    unsigned long long t0, t1;

    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_ILP; i++) {
            asm volatile("fma.rn.f32 %0, %8, %9, %0;"
                         "fma.rn.f32 %1, %8, %9, %1;"
                         "fma.rn.f32 %2, %8, %9, %2;"
                         "fma.rn.f32 %3, %8, %9, %3;"
                         "fma.rn.f32 %4, %8, %9, %4;"
                         "fma.rn.f32 %5, %8, %9, %5;"
                         "fma.rn.f32 %6, %8, %9, %6;"
                         "fma.rn.f32 %7, %8, %9, %7;"
                         : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3),
                           "+f"(a4), "+f"(a5), "+f"(a6), "+f"(a7)
                         : "f"(b), "f"(c));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER_ILP; o++) {
        #pragma unroll
        for (int i = 0; i < INNER_ILP; i++) {
            asm volatile("fma.rn.f32 %0, %8, %9, %0;"
                         "fma.rn.f32 %1, %8, %9, %1;"
                         "fma.rn.f32 %2, %8, %9, %2;"
                         "fma.rn.f32 %3, %8, %9, %3;"
                         "fma.rn.f32 %4, %8, %9, %4;"
                         "fma.rn.f32 %5, %8, %9, %5;"
                         "fma.rn.f32 %6, %8, %9, %6;"
                         "fma.rn.f32 %7, %8, %9, %7;"
                         : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3),
                           "+f"(a4), "+f"(a5), "+f"(a6), "+f"(a7)
                         : "f"(b), "f"(c));
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    float s = a0+a1+a2+a3+a4+a5+a6+a7;
    if (__float_as_int(s) == 0x7FFFFFFF) R[tid] = s;
    if (tid == 0) T[5] = t1 - t0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("=== FFMA Latency Definitive Measurement ===\n");
    printf("Device: %s\n", prop.name);
    printf("SM compute capability: %d.%d\n", prop.major, prop.minor);
    printf("INNER_LAT=%d OUTER_LAT=%d  INNER_ILP=%d OUTER_ILP=%d\n",
           INNER_LAT, OUTER_LAT, INNER_ILP, OUTER_ILP);
    printf("FMAs per chain per run: %d (all variants)\n", INNER_LAT * OUTER_LAT);

    printf("\n--- Current SM clock ---\n");
    fflush(stdout);
    system("nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader 2>/dev/null | head -1");
    fflush(stdout);

    float* d_result;
    unsigned long long* d_timing;
    const int N_T = 8;
    CHECK_CUDA(cudaMalloc(&d_result, 32 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_timing, N_T * sizeof(unsigned long long)));
    unsigned long long* h_timing = (unsigned long long*)calloc(N_T, sizeof(unsigned long long));

    // Warmup
    printf("\nWarming up (3 rounds)...\n");
    fflush(stdout);
    for (int r = 0; r < 3; r++) {
        kernel_selfop<<<1,32>>>(d_result, d_timing, 1.5f);
        kernel_self_const<<<1,32>>>(d_result, d_timing, 1.5f);
        kernel_diffsrc<<<1,32>>>(d_result, d_timing, 1.5f);
        kernel_ilp2<<<1,32>>>(d_result, d_timing, 1.5f);
        kernel_ilp4<<<1,32>>>(d_result, d_timing, 1.5f);
        kernel_ilp8<<<1,32>>>(d_result, d_timing, 1.5f);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    const int N_REPS = 7;
    unsigned long long tall[6][N_REPS];

    printf("Running %d repetitions...\n", N_REPS);
    fflush(stdout);

    for (int rep = 0; rep < N_REPS; rep++) {
        float iv = 1.5f + rep * 0.001f;
        CHECK_CUDA(cudaMemset(d_timing, 0, N_T * sizeof(unsigned long long)));

        kernel_selfop<<<1,32>>>(d_result, d_timing, iv);     CHECK_CUDA(cudaDeviceSynchronize());
        kernel_self_const<<<1,32>>>(d_result, d_timing, iv); CHECK_CUDA(cudaDeviceSynchronize());
        kernel_diffsrc<<<1,32>>>(d_result, d_timing, iv);    CHECK_CUDA(cudaDeviceSynchronize());
        kernel_ilp2<<<1,32>>>(d_result, d_timing, iv);       CHECK_CUDA(cudaDeviceSynchronize());
        kernel_ilp4<<<1,32>>>(d_result, d_timing, iv);       CHECK_CUDA(cudaDeviceSynchronize());
        kernel_ilp8<<<1,32>>>(d_result, d_timing, iv);       CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_timing, d_timing, N_T * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 6; i++) tall[i][rep] = h_timing[i];
    }

    // Median of 7
    auto median7 = [](unsigned long long* a) -> unsigned long long {
        unsigned long long s[7];
        for (int i = 0; i < 7; i++) s[i] = a[i];
        for (int i = 1; i < 7; i++) {
            unsigned long long k = s[i]; int j = i-1;
            while (j >= 0 && s[j] > k) { s[j+1] = s[j]; j--; }
            s[j+1] = k;
        }
        return s[3];
    };

    unsigned long long med[6];
    for (int i = 0; i < 6; i++) med[i] = median7(tall[i]);

    long long ops = (long long)INNER_LAT * OUTER_LAT;  // per chain
    static const int chains[] = {1, 1, 1, 2, 4, 8};
    const char* names[] = {
        "1-chain self-op  [FFMA Ra,Ra,Ra,RZ]",
        "1-chain self+const [FFMA Ra,Ra,Rb,Rc]",
        "1-chain diff-src [FFMA Ra,Rb,Rc,Ra]",
        "2-chain ILP",
        "4-chain ILP",
        "8-chain ILP"
    };

    printf("\n=== RAW TIMING (clock64 cycles, thread-0 only) ===\n");
    printf("%-40s  %7s  %7s  %7s  %7s  %7s  %7s  %7s   median\n",
           "Variant", "r0", "r1", "r2", "r3", "r4", "r5", "r6");
    for (int i = 0; i < 6; i++) {
        printf("%-40s ", names[i]);
        for (int r = 0; r < N_REPS; r++) printf("%7llu  ", tall[i][r]);
        printf("  %7llu\n", med[i]);
    }

    printf("\n=== CYCLES PER FMA (%lld FMAs per chain) ===\n", ops);
    double cy[6];
    for (int i = 0; i < 6; i++)
        cy[i] = (double)med[i] / (double)(ops * chains[i]);

    printf("\nLATENCY-BOUND (single dep chain):\n");
    printf("  V1 self-op   (Ra,Ra,Ra,RZ):    %7.3f cy/FMA\n", cy[0]);
    printf("  V2 self+const (Ra,Ra,Rb,Rc):   %7.3f cy/FMA\n", cy[1]);
    printf("  V3 diff-src  (Ra,Rb,Rc,Ra):    %7.3f cy/FMA  <-- TRUE latency\n", cy[2]);
    printf("\nTHROUGHPUT-APPROACHING (N parallel dep chains):\n");
    printf("  2-chain ILP:  %7.3f cy/FMA\n", cy[3]);
    printf("  4-chain ILP:  %7.3f cy/FMA\n", cy[4]);
    printf("  8-chain ILP:  %7.3f cy/FMA  <-- approaching throughput limit\n", cy[5]);

    double lat = cy[2];
    double tp  = cy[5];
    double inflation_selfop    = cy[0] / lat;
    double inflation_selfconst = cy[1] / lat;

    printf("\nANALYSIS:\n");
    printf("  Self-op inflation vs diff-src:    %.3fx  (port-pressure on 3 same-reg reads)\n",
           inflation_selfop);
    printf("  Self+const inflation vs diff-src: %.3fx\n", inflation_selfconst);
    printf("\n  TRUE FFMA LATENCY:  %.2f cy   (V3 diff-src, cleanest RAW dep chain)\n", lat);
    printf("  PEAK THROUGHPUT:    %.2f cy/FMA (8-chain ILP, B300 single warp)\n", tp);

    // Expected for B300 with dual-issue FMA pipes:
    // 1 SMSP can issue 1 FMA/cy on heavy + 1 FMA/cy on lite = 2 FMA/cy
    // Single warp = 1/4 occupancy of 4 SMSPs → effective 0.5 cy/FMA at full throughput
    // But 8-chain ILP per thread × 32 threads / 4 SMSPs (8 threads per SMSP) = 8 ops in flight
    // Each SMSP handles 8 chains × 8 threads-per-SMSP = 64 ops in flight per SMSP
    // At 4 cy latency and 2-pipe throughput: need ≥ 4/2 = 2 chains to saturate 1 pipe
    // → 8 chains should fully saturate both heavy+lite pipes
    printf("\n  Theoretical for B300 (4 SMSPs, dual FMA pipe, 4 cy lat, 1 cy/pipe throughput):\n");
    printf("    Latency bound (1 chain): 4 cy/FMA\n");
    printf("    Throughput bound (ILP>=4 per SMSP): ~2 cy/FMA (2 chains per pipe) → ~1 cy/FMA total\n");
    printf("    Measured vs theory: latency=%.2f cy (%.1f%% of 4), throughput=%.2f cy (%.1f%% of 1)\n",
           lat, lat / 4.0 * 100.0, tp, tp / 1.0 * 100.0);

    // Chip-wide implications at 2032 MHz
    // Each SMSP dispatches 8 threads to each pipe
    // At throughput tp cy/FMA per thread: chip_TFLOPS = 148 SMs * 4 SMSP * 32 threads * (1/tp) FMA/cy * 2032e6 * 2 FLOP/FMA / 1e12
    double tflops_lat = 148.0 * 4 * 32 * (1.0/lat) * 2032e6 * 2 / 1e12;
    double tflops_tp  = 148.0 * 4 * 32 * (1.0/tp)  * 2032e6 * 2 / 1e12;
    printf("\n  Chip-wide TFLOPS projection (148 SMs, 4 SMSP, 32 threads/warp, 2032 MHz):\n");
    printf("    At latency limit (1 warp, 1 chain):   %.1f TFLOPS\n", tflops_lat);
    printf("    At throughput limit (ILP=8 chains):   %.1f TFLOPS  (ncu-measured peak ~72)\n", tflops_tp);

    printf("\n--- SM clock at end ---\n");
    fflush(stdout);
    system("nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader 2>/dev/null | head -1");
    fflush(stdout);

    cudaFree(d_result);
    cudaFree(d_timing);
    free(h_timing);
    return 0;
}
