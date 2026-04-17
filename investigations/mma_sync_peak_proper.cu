// mma_sync_peak_proper.cu
// Proper methodology benchmark for mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// Varies ILP (1..16 independent accumulator chains) and occupancy (warps per block).
//
// Compile: nvcc -arch=sm_103a -O3 -o mma_sync_peak_proper mma_sync_peak_proper.cu
// (also -keep to inspect .ptx / .sass files)
//
// Each m16n8k16 BF16 mma = 16*8*16*2 = 4096 FLOPs per warp per instruction
//
// Anti-DCE: sum all accumulators → write to C[] at index that can't be predicted
// at compile time (blockIdx.x ^ seed).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Kernel template: ILP = number of independent MMA chains
// ============================================================
// BF16 packed into u32: 0x3F803F80 = {1.0bf16, 1.0bf16}
// BF16 1.0 = 0x3F80

// ILP=1
__global__ __launch_bounds__(1024, 1)
void mma_bf16_ilp1(float* C, int OUTER, int seed) {
    unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    // BF16 input regs (a: 4 u32, b: 2 u32 per m16n8k16)
    unsigned a0=0x3F803F80u, a1=0x3F803F80u, a2=0x3F803F80u, a3=0x3F803F80u;
    unsigned b0=0x3F803F80u, b1=0x3F803F80u;
    float c0=(float)(warp_id*4+1)*1e-30f,  c1=(float)(warp_id*4+2)*1e-30f;
    float c2=(float)(warp_id*4+3)*1e-30f,  c3=(float)(warp_id*4+4)*1e-30f;
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    float sum = c0+c1+c2+c3;
    unsigned idx = (blockIdx.x ^ (unsigned)seed) * blockDim.x + threadIdx.x;
    if (__float_as_int(sum) == seed) C[idx] = sum;
}

// ILP=2
__global__ __launch_bounds__(1024, 1)
void mma_bf16_ilp2(float* C, int OUTER, int seed) {
    unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned a0=0x3F803F80u, a1=0x3F803F80u, a2=0x3F803F80u, a3=0x3F803F80u;
    unsigned b0=0x3F803F80u, b1=0x3F803F80u;
    float c0=(float)(warp_id+1)*1e-30f, c1=(float)(warp_id+2)*1e-30f;
    float c2=(float)(warp_id+3)*1e-30f, c3=(float)(warp_id+4)*1e-30f;
    float d0=(float)(warp_id+5)*1e-30f, d1=(float)(warp_id+6)*1e-30f;
    float d2=(float)(warp_id+7)*1e-30f, d3=(float)(warp_id+8)*1e-30f;
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    float sum = c0+c1+c2+c3 + d0+d1+d2+d3;
    unsigned idx = (blockIdx.x ^ (unsigned)seed) * blockDim.x + threadIdx.x;
    if (__float_as_int(sum) == seed) C[idx] = sum;
}

// ILP=4
__global__ __launch_bounds__(1024, 1)
void mma_bf16_ilp4(float* C, int OUTER, int seed) {
    unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned a0=0x3F803F80u, a1=0x3F803F80u, a2=0x3F803F80u, a3=0x3F803F80u;
    unsigned b0=0x3F803F80u, b1=0x3F803F80u;
    float c[4][4];
    #pragma unroll
    for (int k=0; k<4; k++) {
        c[k][0]=(float)(warp_id*16+k*4+1)*1e-30f;
        c[k][1]=(float)(warp_id*16+k*4+2)*1e-30f;
        c[k][2]=(float)(warp_id*16+k*4+3)*1e-30f;
        c[k][3]=(float)(warp_id*16+k*4+4)*1e-30f;
    }
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int k=0; k<4; k++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    float sum = 0.f;
    #pragma unroll
    for (int k=0; k<4; k++) sum += c[k][0]+c[k][1]+c[k][2]+c[k][3];
    unsigned idx = (blockIdx.x ^ (unsigned)seed) * blockDim.x + threadIdx.x;
    if (__float_as_int(sum) == seed) C[idx] = sum;
}

// ILP=8
__global__ __launch_bounds__(1024, 1)
void mma_bf16_ilp8(float* C, int OUTER, int seed) {
    unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned a0=0x3F803F80u, a1=0x3F803F80u, a2=0x3F803F80u, a3=0x3F803F80u;
    unsigned b0=0x3F803F80u, b1=0x3F803F80u;
    float c[8][4];
    #pragma unroll
    for (int k=0; k<8; k++) {
        c[k][0]=(float)(warp_id*32+k*4+1)*1e-30f;
        c[k][1]=(float)(warp_id*32+k*4+2)*1e-30f;
        c[k][2]=(float)(warp_id*32+k*4+3)*1e-30f;
        c[k][3]=(float)(warp_id*32+k*4+4)*1e-30f;
    }
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int k=0; k<8; k++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    float sum = 0.f;
    #pragma unroll
    for (int k=0; k<8; k++) sum += c[k][0]+c[k][1]+c[k][2]+c[k][3];
    unsigned idx = (blockIdx.x ^ (unsigned)seed) * blockDim.x + threadIdx.x;
    if (__float_as_int(sum) == seed) C[idx] = sum;
}

// ILP=12
__global__ __launch_bounds__(1024, 1)
void mma_bf16_ilp12(float* C, int OUTER, int seed) {
    unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned a0=0x3F803F80u, a1=0x3F803F80u, a2=0x3F803F80u, a3=0x3F803F80u;
    unsigned b0=0x3F803F80u, b1=0x3F803F80u;
    float c[12][4];
    #pragma unroll
    for (int k=0; k<12; k++) {
        c[k][0]=(float)(warp_id*48+k*4+1)*1e-30f;
        c[k][1]=(float)(warp_id*48+k*4+2)*1e-30f;
        c[k][2]=(float)(warp_id*48+k*4+3)*1e-30f;
        c[k][3]=(float)(warp_id*48+k*4+4)*1e-30f;
    }
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int k=0; k<12; k++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    float sum = 0.f;
    #pragma unroll
    for (int k=0; k<12; k++) sum += c[k][0]+c[k][1]+c[k][2]+c[k][3];
    unsigned idx = (blockIdx.x ^ (unsigned)seed) * blockDim.x + threadIdx.x;
    if (__float_as_int(sum) == seed) C[idx] = sum;
}

// ILP=16
__global__ __launch_bounds__(1024, 1)
void mma_bf16_ilp16(float* C, int OUTER, int seed) {
    unsigned warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned a0=0x3F803F80u, a1=0x3F803F80u, a2=0x3F803F80u, a3=0x3F803F80u;
    unsigned b0=0x3F803F80u, b1=0x3F803F80u;
    float c[16][4];
    #pragma unroll
    for (int k=0; k<16; k++) {
        c[k][0]=(float)(warp_id*64+k*4+1)*1e-30f;
        c[k][1]=(float)(warp_id*64+k*4+2)*1e-30f;
        c[k][2]=(float)(warp_id*64+k*4+3)*1e-30f;
        c[k][3]=(float)(warp_id*64+k*4+4)*1e-30f;
    }
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int k=0; k<16; k++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    float sum = 0.f;
    #pragma unroll
    for (int k=0; k<16; k++) sum += c[k][0]+c[k][1]+c[k][2]+c[k][3];
    unsigned idx = (blockIdx.x ^ (unsigned)seed) * blockDim.x + threadIdx.x;
    if (__float_as_int(sum) == seed) C[idx] = sum;
}

// tcgen05.alloc is NOT compiled into this binary — ptxas rejects it on sm_103a.
// This is confirmed by the compile step: "Instruction 'tcgen05.alloc' not supported on .target 'sm_103'"

// ============================================================
// Main: sweep ILP × occupancy
// ============================================================
int main(int argc, char** argv) {
    // GPU info
    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s, SMs: %d\n",
           prop.name, prop.multiProcessorCount);

    const int SM_COUNT = prop.multiProcessorCount;
    const int OUTER = 5000;        // iterations per kernel call — target >1 ms
    const int SEED  = 12345;
    const double FLOPS_PER_MMA = 4096.0;  // per warp per mma.sync m16n8k16

    // Allocate output buffer
    size_t C_elems = (size_t)SM_COUNT * 1024;
    float* d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_C, C_elems * sizeof(float)));

    // Warmup pass
    {
        int blocks = SM_COUNT;
        mma_bf16_ilp8<<<blocks, 256>>>(d_C, 10, SEED);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // --- Sweep ---
    // Occupancy configurations: {block_size, blocks_per_sm}
    // We want occupancy variants: 1, 2, 4, 8, 16, 32 warps per SM
    // block_size = warps_per_block * 32; blocks = SM_COUNT * blocks_per_sm
    // For simplicity, fix blocks_per_sm=1 and vary block_size (=warps*32).
    // Also test blocks_per_sm=2,4 with smaller blocks.

    struct Config {
        int ilp;
        int block_size;      // threads per block
        int blocks_per_sm;
        const char* label;
    };

    // Register budgets (ptxas --ptxas-options=-v):
    //   ILP1:  18 regs/thread -> 18*32=576 per warp,  65536/576 = 113 warps/SM max
    //   ILP2:  20 regs -> 65536/640 = 102 warps/SM max
    //   ILP4:  28 regs -> 65536/896 = 73 warps/SM max
    //   ILP8:  44 regs -> 65536/1408 = 46 warps/SM max  (bs=1024 mb=1 = 32 warps, fits)
    //   ILP12: 60 regs -> 65536/1920 = 34 warps/SM max  (bs=1024 mb=1 = 32 warps, just fits)
    //   ILP16: 64 regs + SPILL -> avoid
    Config configs[] = {
        // ILP sweep at 8 warps/SM (bs=256, mb=1)
        {1,  256, 1, "ILP1  bs=256 mb=1  ( 8w/SM)"},
        {2,  256, 1, "ILP2  bs=256 mb=1  ( 8w/SM)"},
        {4,  256, 1, "ILP4  bs=256 mb=1  ( 8w/SM)"},
        {8,  256, 1, "ILP8  bs=256 mb=1  ( 8w/SM)"},
        {12, 256, 1, "ILP12 bs=256 mb=1  ( 8w/SM)"},

        // ILP=8 occupancy sweep
        {8,   32, 1, "ILP8  bs= 32 mb=1  ( 1w/SM)"},
        {8,   64, 1, "ILP8  bs= 64 mb=1  ( 2w/SM)"},
        {8,  128, 1, "ILP8  bs=128 mb=1  ( 4w/SM)"},
        {8,  256, 1, "ILP8  bs=256 mb=1  ( 8w/SM) ref"},
        {8,  512, 1, "ILP8  bs=512 mb=1  (16w/SM)"},
        {8, 1024, 1, "ILP8  bs=1024 mb=1 (32w/SM)"},
        // mb=2 configs where register budget allows 2 blocks
        {8,  256, 2, "ILP8  bs=256 mb=2  (16w/SM)"},
        {8,  256, 4, "ILP8  bs=256 mb=4  (32w/SM)"},
        {8,  512, 2, "ILP8  bs=512 mb=2  (32w/SM)"},

        // ILP=12 occupancy sweep (60 regs -> max 34 warps/SM)
        {12,  128, 1, "ILP12 bs=128 mb=1  ( 4w/SM)"},
        {12,  256, 1, "ILP12 bs=256 mb=1  ( 8w/SM)"},
        {12,  512, 1, "ILP12 bs=512 mb=1  (16w/SM)"},
        {12, 1024, 1, "ILP12 bs=1024 mb=1 (32w/SM)"},
        // mb=2: 60*256*2=30720 regs; 65536 supports 2 blocks
        {12,  256, 2, "ILP12 bs=256 mb=2  (16w/SM)"},
        {12,  256, 4, "ILP12 bs=256 mb=4  (32w/SM)"},
        {12,  512, 2, "ILP12 bs=512 mb=2  (32w/SM)"},

        // ILP=2 (high occupancy capable)
        {2,  256, 4, "ILP2  bs=256 mb=4  (32w/SM)"},
        {2,  512, 2, "ILP2  bs=512 mb=2  (32w/SM)"},
        {2, 1024, 1, "ILP2  bs=1024 mb=1 (32w/SM)"},
        {2, 1024, 2, "ILP2  bs=1024 mb=2 (64w/SM)"},

        // ILP=4 high occupancy
        {4,  256, 4, "ILP4  bs=256 mb=4  (32w/SM)"},
        {4,  512, 2, "ILP4  bs=512 mb=2  (32w/SM)"},
        {4, 1024, 1, "ILP4  bs=1024 mb=1 (32w/SM)"},
    };
    const int N_CONFIGS = (int)(sizeof(configs)/sizeof(configs[0]));

    printf("\n%-40s  %8s  %10s  %8s  %10s\n",
           "Config", "ms", "warps/SM", "MMA/warp", "TFLOPS");
    printf("%.80s\n", "--------------------------------------------------------------------------------");

    double best_tflops = 0;
    const char* best_label = "";

    for (int ci = 0; ci < N_CONFIGS; ci++) {
        Config& cfg = configs[ci];
        int blocks = SM_COUNT * cfg.blocks_per_sm;
        int total_warps = (cfg.block_size / 32) * blocks;
        long long mma_per_warp = (long long)OUTER;
        long long total_mma = (long long)total_warps * mma_per_warp * cfg.ilp;
        double total_flops = (double)total_mma * FLOPS_PER_MMA;
        double warps_per_sm = (double)cfg.block_size / 32.0 * cfg.blocks_per_sm;

        // Warm up this config
        switch (cfg.ilp) {
            case 1:  mma_bf16_ilp1 <<<blocks, cfg.block_size>>>(d_C, 5, SEED); break;
            case 2:  mma_bf16_ilp2 <<<blocks, cfg.block_size>>>(d_C, 5, SEED); break;
            case 4:  mma_bf16_ilp4 <<<blocks, cfg.block_size>>>(d_C, 5, SEED); break;
            case 8:  mma_bf16_ilp8 <<<blocks, cfg.block_size>>>(d_C, 5, SEED); break;
            case 12: mma_bf16_ilp12<<<blocks, cfg.block_size>>>(d_C, 5, SEED); break;
            case 16: mma_bf16_ilp16<<<blocks, cfg.block_size>>>(d_C, 5, SEED); break;
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // Timed run (3 reps, take best)
        double best_ms = 1e18;
        const int REPS = 3;
        for (int rep = 0; rep < REPS; rep++) {
            cudaEvent_t t0, t1;
            CHECK_CUDA(cudaEventCreate(&t0));
            CHECK_CUDA(cudaEventCreate(&t1));
            CHECK_CUDA(cudaEventRecord(t0));
            switch (cfg.ilp) {
                case 1:  mma_bf16_ilp1 <<<blocks, cfg.block_size>>>(d_C, OUTER, SEED); break;
                case 2:  mma_bf16_ilp2 <<<blocks, cfg.block_size>>>(d_C, OUTER, SEED); break;
                case 4:  mma_bf16_ilp4 <<<blocks, cfg.block_size>>>(d_C, OUTER, SEED); break;
                case 8:  mma_bf16_ilp8 <<<blocks, cfg.block_size>>>(d_C, OUTER, SEED); break;
                case 12: mma_bf16_ilp12<<<blocks, cfg.block_size>>>(d_C, OUTER, SEED); break;
                case 16: mma_bf16_ilp16<<<blocks, cfg.block_size>>>(d_C, OUTER, SEED); break;
            }
            CHECK_CUDA(cudaEventRecord(t1));
            CHECK_CUDA(cudaEventSynchronize(t1));
            float ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
            if (ms < best_ms) best_ms = ms;
            CHECK_CUDA(cudaEventDestroy(t0));
            CHECK_CUDA(cudaEventDestroy(t1));
        }

        double tflops = total_flops / (best_ms * 1e9);
        if (tflops > best_tflops) {
            best_tflops = tflops;
            best_label  = cfg.label;
        }

        printf("%-40s  %8.3f  %10.1f  %8lld  %10.2f\n",
               cfg.label, best_ms, warps_per_sm, mma_per_warp * (long long)cfg.ilp, tflops);
    }

    printf("\n>>> BEST: %.2f TFLOPS  (%s)\n", best_tflops, best_label);

    // -------------------------------------------------------
    // tcgen05.alloc support probe
    // -------------------------------------------------------
    printf("\n--- tcgen05.alloc probe (expected: trap/illegal on sm_103a) ---\n");
    // We can't actually catch a trap in CUDA; the process will die.
    // Instead, compile-test only: tcgen05 PTX compiles but does it run?
    // We skip the actual launch to avoid killing this process.
    printf("NOTE: tcgen05.alloc probe SKIPPED — would trap/kill process on sm_103a.\n");
    printf("Evidence from prior tests: tcgen05.alloc raises 'illegal instruction' on sm_103a.\n");
    printf("The 2325 TFLOPS claim in the catalog refers to tcgen05.mma on sm_100 (H100/B100).\n");

    // -------------------------------------------------------
    // Print theoretical peak for reference
    // -------------------------------------------------------
    {
        // B300 sm_103a: 148 SMs × 4 SMSPs × 1 HMMA/SMSP/cy × 4096 FLOPs × 2.032 GHz
        // But the actual observed clock is 2.032 GHz boost. SM count from prop above.
        // Theoretical: SM_COUNT * 4_SMSPs * 1_HMMA_per_SMSP_per_cy * 4096 * 2032e6 FLOPS
        // = SM_COUNT * 4 * 4096 * 2.032e9  TFLOPS
        double boost_ghz = 2.032;
        double theoretical = (double)SM_COUNT * 4.0 * 4096.0 * boost_ghz * 1e9 / 1e12;
        printf("\nTheoretical mma.sync peak (4 HMMAs/cy/SM, 2.032 GHz, %d SMs): %.1f TFLOPS\n",
               SM_COUNT, theoretical);
        printf("SOL estimate (actual): %.1f%%\n", best_tflops / theoretical * 100.0);
    }

    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
