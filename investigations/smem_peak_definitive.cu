// smem_peak_definitive.cu — Settle the SHMEM BW dispute on B300
//
// Tests:
//   T1: ld.shared non-volatile  (reproduces AUDIT_NOTES ~19.85 TB/s methodology)
//   T2: ld.volatile.shared.v4   (reproduces PIPE_CATALOG ~35 TB/s methodology)
//   T3: ld.volatile.shared.u32 ×4, stride-32 (conflict-free)
//   T4: volatile v4, UNROLL=32 inner loop (catalog-exact)
//   T5: ldmatrix.x4 tensor instruction
//   T6: float4 non-volatile vector load
//   T7: volatile v4 SMEM size sweep (4KB / 16KB / 56KB / 112KB)
//   T8: non-volatile with true anti-DCE (pointer-chase style)
//
// Anti-DCE discipline:
//   - All sinks are unconditional writes (no `if (x < -1e30f)`)
//   - Loop counter used in address calculation each iteration
//   - `ld.volatile.shared` forces emission even when ptxas can prove result
//
// Compile: nvcc -arch=sm_103a -O3 -std=c++17 smem_peak_definitive.cu -o smem_peak_definitive
// Pin clock: nvidia-smi -lgc 2032 (before running)

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

// =====================================================================
// T1: Plain ld.shared (non-volatile)
// Exact analog of tests/shmem_peak.cu — 8 scalar float loads per iter
// Uses `(i * 256) & mask` base — compiler CAN analyze but may still DCE
// =====================================================================
__global__ void __launch_bounds__(1024, 2)
t1_ld_shared_nonvol(float *out, int iters, int smem_floats)
{
    extern __shared__ float smem[];
    for (int i = threadIdx.x; i < smem_floats; i += blockDim.x)
        smem[i] = (float)(i & 0x3ff);
    __syncthreads();

    int tid  = threadIdx.x;
    int lane = tid & 31;

    float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
    int mask = smem_floats - 1;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int base = (i * 256) & mask;
        a0 += smem[(base +   0 + lane) & mask];
        a1 += smem[(base +  32 + lane) & mask];
        a2 += smem[(base +  64 + lane) & mask];
        a3 += smem[(base +  96 + lane) & mask];
        a4 += smem[(base + 128 + lane) & mask];
        a5 += smem[(base + 160 + lane) & mask];
        a6 += smem[(base + 192 + lane) & mask];
        a7 += smem[(base + 224 + lane) & mask];
    }

    // Unconditional sink — forces all accumulators live
    if (tid == 0 && blockIdx.x == 0)
        out[0] = a0+a1+a2+a3+a4+a5+a6+a7;
    else if (tid < 8)
        out[tid] = (&a0)[0];  // prevent compiler merging the 8 values
}

// =====================================================================
// T2: ld.volatile.shared.v4.u32
// Forces compiler to emit each load in the loop body.
// 2 v4 loads per iter = 8 u32 = 32 bytes per thread per iter.
// =====================================================================
__global__ void __launch_bounds__(1024, 2)
t2_ldvol_v4(unsigned int *out, int iters, int smem_u32s)
{
    extern __shared__ unsigned int smem2[];
    for (int i = threadIdx.x; i < smem_u32s; i += blockDim.x)
        smem2[i] = (unsigned int)i;
    __syncthreads();

    int tid  = threadIdx.x;
    int mask = smem_u32s - 1;

    unsigned int a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int base = (i * 4) & mask;
        unsigned int lv0,lv1,lv2,lv3;

        // First v4 at base + tid*4
        unsigned int addr1 = (unsigned int)__cvta_generic_to_shared(&smem2[(base + tid*4) & mask]);
        asm volatile(
            "ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(lv0),"=r"(lv1),"=r"(lv2),"=r"(lv3)
            : "r"(addr1));
        a0 += lv0; a1 += lv1; a2 += lv2; a3 += lv3;

        // Second v4 at base + smem_u32s/2 + tid*4
        unsigned int addr2 = (unsigned int)__cvta_generic_to_shared(&smem2[(base + smem_u32s/2 + tid*4) & mask]);
        asm volatile(
            "ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(lv0),"=r"(lv1),"=r"(lv2),"=r"(lv3)
            : "r"(addr2));
        a4 += lv0; a5 += lv1; a6 += lv2; a7 += lv3;
    }

    if (tid == 0 && blockIdx.x == 0)
        out[0] = a0+a1+a2+a3+a4+a5+a6+a7;
}

// =====================================================================
// T3: 4 volatile u32 loads, stride-32 (conflict-free)
// Thread t in a warp: loads smem[base+t], smem[base+t+32],
//                           smem[base+t+64], smem[base+t+96]
// Each element of a warp hits a UNIQUE bank (32 threads × unique bank).
// =====================================================================
__global__ void __launch_bounds__(1024, 2)
t3_ldvol_u32x4_stride32(unsigned int *out, int iters)
{
    extern __shared__ unsigned int smem3[];
    const int SMEM_U32 = 12288;  // 48 KB

    for (int i = threadIdx.x; i < SMEM_U32; i += blockDim.x)
        smem3[i] = (unsigned int)i;
    __syncthreads();

    int tid  = threadIdx.x;
    int mask = SMEM_U32 - 1;

    unsigned int a0=0, a1=0, a2=0, a3=0;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int base = (i * 128) & mask;  // shift by 128 floats each iter
        unsigned int lv0, lv1, lv2, lv3;

        unsigned int a3_0 = (unsigned int)__cvta_generic_to_shared(&smem3[(base + tid     ) & mask]);
        unsigned int a3_1 = (unsigned int)__cvta_generic_to_shared(&smem3[(base + tid + 32) & mask]);
        unsigned int a3_2 = (unsigned int)__cvta_generic_to_shared(&smem3[(base + tid + 64) & mask]);
        unsigned int a3_3 = (unsigned int)__cvta_generic_to_shared(&smem3[(base + tid + 96) & mask]);
        asm volatile("ld.volatile.shared.u32 %0, [%1];"
            : "=r"(lv0) : "r"(a3_0));
        asm volatile("ld.volatile.shared.u32 %0, [%1];"
            : "=r"(lv1) : "r"(a3_1));
        asm volatile("ld.volatile.shared.u32 %0, [%1];"
            : "=r"(lv2) : "r"(a3_2));
        asm volatile("ld.volatile.shared.u32 %0, [%1];"
            : "=r"(lv3) : "r"(a3_3));
        a0 += lv0; a1 += lv1; a2 += lv2; a3 += lv3;
    }

    if (tid == 0 && blockIdx.x == 0) out[0] = a0+a1+a2+a3;
}

// =====================================================================
// T4: Catalog-exact methodology
// UNROLL=32 inner v4 loads per outer iter.  Each unroll step strides
// by smem_u32s/32 to spread across the smem array.
// bs=1024, mb=2, ITERS outer × 32 inner = 32× more loads per outer iter.
// =====================================================================
__global__ void __launch_bounds__(1024, 2)
t4_catalog_exact(unsigned int *out, int iters)
{
    extern __shared__ unsigned int smem4[];
    const int SMEM_U32 = 14336;  // 56 KB = 14336 u32

    for (int i = threadIdx.x; i < SMEM_U32; i += blockDim.x)
        smem4[i] = (unsigned int)i;
    __syncthreads();

    int tid  = threadIdx.x;
    int mask = SMEM_U32 - 1;

    unsigned int a0=0, a1=0, a2=0, a3=0;
    const int STRIDE = SMEM_U32 / 32;  // step per unroll, spans full smem

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int base = (i * 4) & mask;  // outer shift for anti-DCE
        unsigned int lv0, lv1, lv2, lv3;

        #pragma unroll 32
        for (int u = 0; u < 32; u++) {
            int aidx = (base + tid*4 + u*STRIDE) & mask;
            unsigned int aptr = (unsigned int)__cvta_generic_to_shared(&smem4[aidx]);
            asm volatile(
                "ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(lv0),"=r"(lv1),"=r"(lv2),"=r"(lv3)
                : "r"(aptr));
            a0 += lv0; a1 += lv1; a2 += lv2; a3 += lv3;
        }
    }

    if (tid == 0 && blockIdx.x == 0) out[0] = a0+a1+a2+a3;
}

// =====================================================================
// T5: ldmatrix.x4.m8n8.shared.b16
// Each warp reads 8 rows × 8 cols × 2 bytes = 128 bytes per thread
// = 4096 bytes per warp.  Thread t provides the row address;
// HW transposes and distributes across the warp.
// =====================================================================
__global__ void __launch_bounds__(1024, 2)
t5_ldmatrix_x4(unsigned int *out, int iters)
{
    extern __shared__ unsigned short smem5[];
    const int SMEM_U16 = 28672;  // 56 KB / 2

    for (int i = threadIdx.x; i < SMEM_U16; i += blockDim.x)
        smem5[i] = (unsigned short)(i & 0xffff);
    __syncthreads();

    int tid  = threadIdx.x;
    int lane = tid & 31;
    unsigned int a0=0, a1=0, a2=0, a3=0;

    // Each thread provides 1 row address; 8 threads per m8 group provide 8 rows.
    // Row size = 16 bytes = 8 u16.  Stride 16 bytes × 32 threads = 512 bytes/warp/iter.
    int smem_bytes = SMEM_U16 * 2;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // Address for this thread's row — shift per outer iter
        int row_byte = ((i * 512 + lane * 16) % smem_bytes);
        unsigned int lv0, lv1, lv2, lv3;
        unsigned int lm_addr = (unsigned int)__cvta_generic_to_shared((char*)smem5 + row_byte);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(lv0),"=r"(lv1),"=r"(lv2),"=r"(lv3)
            : "r"(lm_addr));
        a0 += lv0; a1 += lv1; a2 += lv2; a3 += lv3;
    }

    if (tid == 0 && blockIdx.x == 0) out[0] = a0+a1+a2+a3;
}

// =====================================================================
// T6: float4 non-volatile (may be hoisted by ptxas)
// =====================================================================
__global__ void __launch_bounds__(1024, 2)
t6_float4_nonvol(float *out, int iters)
{
    extern __shared__ float4 smem6[];
    const int SMEM_F4 = 3072;  // 48 KB / 16

    for (int i = threadIdx.x; i < SMEM_F4 * 4; i += blockDim.x)
        ((float*)smem6)[i] = (float)(i & 0x3ff);
    __syncthreads();

    int tid  = threadIdx.x;
    int mask = SMEM_F4 - 1;
    float a0=0, a1=0, a2=0, a3=0;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int addr = (i * 32 + tid) & mask;
        float4 v = smem6[addr];
        a0 += v.x; a1 += v.y; a2 += v.z; a3 += v.w;
    }

    if (tid == 0 && blockIdx.x == 0) out[0] = a0+a1+a2+a3;
}

// =====================================================================
// T7: SMEM size sweep — volatile v4, configurable size
// =====================================================================
__global__ void __launch_bounds__(1024, 2)
t7_size_sweep(unsigned int *out, int iters, int smem_u32s)
{
    extern __shared__ unsigned int smem7[];
    for (int i = threadIdx.x; i < smem_u32s; i += blockDim.x)
        smem7[i] = (unsigned int)i;
    __syncthreads();

    int tid  = threadIdx.x;
    int mask = smem_u32s - 1;
    unsigned int a0=0, a1=0, a2=0, a3=0;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int base = (i * 4) & mask;
        unsigned int lv0, lv1, lv2, lv3;
        unsigned int t7_ptr = (unsigned int)__cvta_generic_to_shared(&smem7[(base + tid*4) & mask]);
        asm volatile(
            "ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(lv0),"=r"(lv1),"=r"(lv2),"=r"(lv3)
            : "r"(t7_ptr));
        a0 += lv0; a1 += lv1; a2 += lv2; a3 += lv3;
    }

    if (tid == 0 && blockIdx.x == 0) out[0] = a0+a1+a2+a3;
}

// =====================================================================
// T8: Non-volatile loads but with stronger anti-DCE: use ALL 8 results
// as independent accumulator chains with no cross-iter dependency on base
// (tests whether ptxas still folds despite unpredictable chain)
// =====================================================================
__global__ void __launch_bounds__(1024, 2)
t8_nonvol_strong_antidce(float *out, int iters)
{
    extern __shared__ float smem8[];
    const int SMEM_F = 14336;  // 56 KB / 4

    for (int i = threadIdx.x; i < SMEM_F; i += blockDim.x)
        smem8[i] = (float)(i & 0x3ff) + 0.5f;
    __syncthreads();

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int mask = SMEM_F - 1;

    // 8 independent accumulator chains, each with unique non-linear base
    float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // XOR with iter index to defeat stride-analysis
        int base = ((i ^ (i >> 4)) * 128) & mask;
        a0 += smem8[(base +   0 + lane) & mask];
        a1 += smem8[(base +  32 + lane) & mask];
        a2 += smem8[(base +  64 + lane) & mask];
        a3 += smem8[(base +  96 + lane) & mask];
        a4 += smem8[(base + 128 + lane) & mask];
        a5 += smem8[(base + 160 + lane) & mask];
        a6 += smem8[(base + 192 + lane) & mask];
        a7 += smem8[(base + 224 + lane) & mask];
    }

    // All 8 accumulators unconditionally written
    out[tid * 8 + 0] = a0; out[tid * 8 + 1] = a1;
    out[tid * 8 + 2] = a2; out[tid * 8 + 3] = a3;
    out[tid * 8 + 4] = a4; out[tid * 8 + 5] = a5;
    out[tid * 8 + 6] = a6; out[tid * 8 + 7] = a7;
}

// =====================================================================
// CUDA event timing helper
// =====================================================================
template<typename F>
static float time_kernel_ms(int nwarmup, int ntrials, F fn)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    for (int i = 0; i < nwarmup; i++) { fn(); cudaDeviceSynchronize(); }

    float best = 1e30f;
    for (int i = 0; i < ntrials; i++) {
        cudaEventRecord(e0);
        fn();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return best;
}

static float bw_TBs(long long bytes, float ms) {
    return (double)bytes / (ms / 1000.0) / 1e12;
}

int main()
{
    int dev = 0;
    cudaSetDevice(dev);

    int sm;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
    int clk_khz;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev);
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    double clk_ghz   = clk_khz / 1e6;
    double theory_tbps = 32.0 * 4.0 * clk_ghz * sm / 1e3;  // TB/s
    double theory_gbs  = 128.0 * clk_ghz;                   // GB/s per SM

    char devname[256] = "";
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        snprintf(devname, sizeof(devname), "%s", prop.name);
    }

    printf("=== B300 SHMEM BW Definitive Test ===\n");
    printf("Device: %s, %d SMs\n", devname, sm);
    printf("Clock: %.3f GHz\n", clk_ghz);
    printf("Max smem/block (opt-in): %d KB\n", max_smem / 1024);
    printf("Theoretical: 32 banks × 4 B/cy × %.3f GHz × %d SMs = %.2f TB/s\n",
           clk_ghz, sm, theory_tbps);
    printf("Per-SM theoretical: 128 B/cy × %.3f GHz = %.1f GB/s/SM\n\n",
           clk_ghz, theory_gbs);

    const int ITERS  = 2000;
    const int WARMUP = 5;
    const int TRIALS = 20;
    const int BS     = 1024;
    const int MB     = 2;   // max blocks per SM from __launch_bounds__
    int blocks       = sm * MB;

    // Global output buffer (T8 writes BS*8 floats per block — just use GPU 0)
    // T8 uses 1 block only, so blocks_t8=1
    float *d_out;
    cudaMalloc(&d_out, (long long)BS * 8 * sizeof(float) + 64);

    // ─────────────────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("T1: ld.shared non-volatile, 8 scalar float/iter, 16 KB smem\n");
    // ─────────────────────────────────────────────────────────────────────
    {
        int smem_bytes  = 16 * 1024;
        int smem_floats = smem_bytes / 4;
        cudaFuncSetAttribute(t1_ld_shared_nonvol,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

        float ms = time_kernel_ms(WARMUP, TRIALS, [&]{
            t1_ld_shared_nonvol<<<blocks, BS, smem_bytes>>>(d_out, ITERS, smem_floats);
        });
        long long bytes = (long long)blocks * BS * ITERS * 8 * 4;
        float tb = bw_TBs(bytes, ms);
        printf("   blocks=%d×%d=%d  iters=%d  time=%.3f ms\n",
               sm, MB, blocks, ITERS, ms);
        printf("   BW: %.2f TB/s  (%.1f GB/s/SM  %.1f%% of theory)\n\n",
               tb, tb*1e3/sm, tb/theory_tbps*100);
    }

    // ─────────────────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("T2: ld.volatile.shared.v4.u32, 2 v4/iter=32B/thread, 56 KB smem\n");
    // ─────────────────────────────────────────────────────────────────────
    {
        int smem_bytes = 56 * 1024;
        int smem_u32s  = smem_bytes / 4;
        cudaFuncSetAttribute(t2_ldvol_v4,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

        float ms = time_kernel_ms(WARMUP, TRIALS, [&]{
            t2_ldvol_v4<<<blocks, BS, smem_bytes>>>((unsigned int*)d_out, ITERS, smem_u32s);
        });
        long long bytes = (long long)blocks * BS * ITERS * 8 * 4;
        float tb = bw_TBs(bytes, ms);
        printf("   blocks=%d  iters=%d  time=%.3f ms\n", blocks, ITERS, ms);
        printf("   BW: %.2f TB/s  (%.1f GB/s/SM  %.1f%% of theory)\n\n",
               tb, tb*1e3/sm, tb/theory_tbps*100);
    }

    // ─────────────────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("T3: volatile u32 ×4, stride-32 conflict-free, 48 KB smem\n");
    // ─────────────────────────────────────────────────────────────────────
    {
        int smem_bytes = 48 * 1024;
        cudaFuncSetAttribute(t3_ldvol_u32x4_stride32,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

        float ms = time_kernel_ms(WARMUP, TRIALS, [&]{
            t3_ldvol_u32x4_stride32<<<blocks, BS, smem_bytes>>>((unsigned int*)d_out, ITERS);
        });
        long long bytes = (long long)blocks * BS * ITERS * 4 * 4;
        float tb = bw_TBs(bytes, ms);
        printf("   blocks=%d  iters=%d  time=%.3f ms\n", blocks, ITERS, ms);
        printf("   BW: %.2f TB/s  (%.1f GB/s/SM  %.1f%% of theory)\n\n",
               tb, tb*1e3/sm, tb/theory_tbps*100);
    }

    // ─────────────────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("T4: catalog-exact: UNROLL=32 × v4 per iter, 56 KB smem\n");
    // ─────────────────────────────────────────────────────────────────────
    {
        int smem_bytes = 56 * 1024;
        cudaFuncSetAttribute(t4_catalog_exact,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

        float ms = time_kernel_ms(WARMUP, TRIALS, [&]{
            t4_catalog_exact<<<blocks, BS, smem_bytes>>>((unsigned int*)d_out, ITERS);
        });
        // 32 v4 per iter = 128 u32 = 512 bytes per thread
        long long bytes = (long long)blocks * BS * ITERS * 32 * 4 * 4;
        float tb = bw_TBs(bytes, ms);
        printf("   blocks=%d  iters=%d  time=%.3f ms\n", blocks, ITERS, ms);
        printf("   BW: %.2f TB/s  (%.1f GB/s/SM  %.1f%% of theory)\n\n",
               tb, tb*1e3/sm, tb/theory_tbps*100);
    }

    // ─────────────────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("T5: ldmatrix.x4 (128b per thread per iter), 56 KB smem\n");
    // ─────────────────────────────────────────────────────────────────────
    {
        int smem_bytes = 56 * 1024;
        cudaFuncSetAttribute(t5_ldmatrix_x4,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

        float ms = time_kernel_ms(WARMUP, TRIALS, [&]{
            t5_ldmatrix_x4<<<blocks, BS, smem_bytes>>>((unsigned int*)d_out, ITERS);
        });
        // ldmatrix.x4 = 128b = 16 bytes per thread per iter
        long long bytes = (long long)blocks * BS * ITERS * 16;
        float tb = bw_TBs(bytes, ms);
        printf("   blocks=%d  iters=%d  time=%.3f ms\n", blocks, ITERS, ms);
        printf("   BW: %.2f TB/s  (%.1f GB/s/SM  %.1f%% of theory)\n\n",
               tb, tb*1e3/sm, tb/theory_tbps*100);
    }

    // ─────────────────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("T6: float4 non-volatile, 48 KB smem\n");
    // ─────────────────────────────────────────────────────────────────────
    {
        int smem_bytes = 48 * 1024;
        cudaFuncSetAttribute(t6_float4_nonvol,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

        float ms = time_kernel_ms(WARMUP, TRIALS, [&]{
            t6_float4_nonvol<<<blocks, BS, smem_bytes>>>(d_out, ITERS);
        });
        long long bytes = (long long)blocks * BS * ITERS * 16;
        float tb = bw_TBs(bytes, ms);
        printf("   blocks=%d  iters=%d  time=%.3f ms\n", blocks, ITERS, ms);
        printf("   BW: %.2f TB/s  (%.1f GB/s/SM  %.1f%% of theory)\n\n",
               tb, tb*1e3/sm, tb/theory_tbps*100);
    }

    // ─────────────────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("T7: SMEM size sweep (volatile v4, 1 v4/iter, 1 or 2 blocks/SM):\n");
    // ─────────────────────────────────────────────────────────────────────
    {
        struct { int kb; const char *label; int mb; } configs[] = {
            { 4,   "  4 KB", 2},
            {16,   " 16 KB", 2},
            {56,   " 56 KB", 2},
            {112,  "112 KB", 1},  // only 1 block/SM — too big for 2
        };
        for (auto &c : configs) {
            int smem_bytes = c.kb * 1024;
            int smem_u32s  = smem_bytes / 4;
            int blks       = sm * c.mb;

            cudaFuncSetAttribute(t7_size_sweep,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

            float ms = time_kernel_ms(WARMUP, TRIALS, [&]{
                t7_size_sweep<<<blks, BS, smem_bytes>>>((unsigned int*)d_out, ITERS, smem_u32s);
            });
            cudaDeviceSynchronize();

            long long bytes = (long long)blks * BS * ITERS * 16;
            float tb = bw_TBs(bytes, ms);
            printf("   smem=%s blocks=%d(%d/SM):  %.2f TB/s  %.1f GB/s/SM  %.1f%%\n",
                   c.label, blks, c.mb, tb, tb*1e3/sm, tb/theory_tbps*100);
        }
        printf("\n");
    }

    // ─────────────────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("T8: non-volatile, stronger anti-DCE (XOR base, all 8 accs written), 56 KB\n");
    // ─────────────────────────────────────────────────────────────────────
    {
        int smem_bytes = 56 * 1024;
        cudaFuncSetAttribute(t8_nonvol_strong_antidce,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

        // T8 writes BS*8 floats per block → allocate enough
        float *d_big;
        cudaMalloc(&d_big, (long long)1 * BS * 8 * sizeof(float));

        float ms = time_kernel_ms(WARMUP, TRIALS, [&]{
            // 1 block only (output buffer sized for 1 block)
            t8_nonvol_strong_antidce<<<1, BS, smem_bytes>>>(d_big, ITERS);
        });
        // Scale to all SMs for comparison
        long long bytes_1blk = (long long)1 * BS * ITERS * 8 * 4;
        float tb_1blk = bw_TBs(bytes_1blk, ms);
        // Extrapolate: if sm blocks ran in same time
        float tb_extrap = tb_1blk * sm;  // hypothetical, not real
        printf("   1 block only (safety), 8 loads/iter\n");
        printf("   time=%.3f ms  1-block BW=%.3f TB/s\n", ms, tb_1blk);
        printf("   Per-SM BW: %.1f GB/s  (extrapolated chip: %.2f TB/s)\n\n",
               tb_1blk*1e3, tb_extrap);

        // Now run with full blocks
        float ms2 = time_kernel_ms(WARMUP, TRIALS, [&]{
            t8_nonvol_strong_antidce<<<blocks, BS, smem_bytes>>>(d_big, ITERS);
        });
        // Each block writes 1024*8 = 8192 floats → d_big too small, just measure time
        long long bytes_full = (long long)blocks * BS * ITERS * 8 * 4;
        float tb_full = bw_TBs(bytes_full, ms2);
        printf("   Full blocks=%d  time=%.3f ms\n", blocks, ms2);
        printf("   BW: %.2f TB/s  (%.1f GB/s/SM  %.1f%%)\n\n",
               tb_full, tb_full*1e3/sm, tb_full/theory_tbps*100);

        cudaFree(d_big);
    }

    // ─────────────────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("SUMMARY TABLE\n");
    printf("Theoretical peak: %.2f TB/s chip, %.1f GB/s/SM at %.3f GHz\n",
           theory_tbps, theory_gbs, clk_ghz);
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("(See per-test output above for detailed numbers)\n\n");

    cudaFree(d_out);
    printf("NOTE: verify SASS load counts with:\n");
    printf("  nvcc -arch=sm_103a -O3 -cubin smem_peak_definitive.cu\n");
    printf("  cuobjdump -sass smem_peak_definitive.cubin | grep -c 'LDS\\|LDSM'\n");

    return 0;
}
