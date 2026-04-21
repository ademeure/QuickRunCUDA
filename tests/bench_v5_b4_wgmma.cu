// V5 B4: WGMMA (Hopper warpgroup MMA) on B300?
// Test if wgmma.mma_async.sync PTX compiles and runs on sm_103a.
// Minimal m64n64k16 BF16 example.
extern "C" __global__ __launch_bounds__(128, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // WGMMA requires:
    // - 128 threads = 1 warpgroup (4 warps)
    // - SMEM operands with specific descriptors

    __shared__ __align__(1024) unsigned int smemA[64 * 16 / 2];  // 64x16 BF16 packed
    __shared__ __align__(1024) unsigned int smemB[64 * 16 / 2];

    if (threadIdx.x == 0) {
        for (int i = 0; i < 64*16/2; i++) {
            smemA[i] = 0x3F803F80;  // BF16(1.0, 1.0)
            smemB[i] = 0x3F803F80;
        }
    }
    __syncthreads();

    // Build descriptors for smemA/smemB
    unsigned long long descA = ((unsigned long long)__cvta_generic_to_shared(smemA) >> 4) & 0x3FFF;
    unsigned long long descB = ((unsigned long long)__cvta_generic_to_shared(smemB) >> 4) & 0x3FFF;
    descA |= (16ULL << 16);  // leading dim byte offset
    descB |= (16ULL << 16);
    descA |= (32ULL << 32);  // stride byte offset
    descB |= (32ULL << 32);

    // m64n64k16 BF16 accumulator in registers (32 floats per thread in this warpgroup)
    float d[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) d[i] = 0.0f;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    asm volatile("wgmma.fence.sync.aligned;");

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
            "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
            " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
            "%32, %33, 1, 1, 1, 0, 0;"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
              "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
              "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
              "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
              "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
              "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
              "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
              "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
            : "l"(descA), "l"(descB));
    }

    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Anti-DCE
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) sum += d[i];
    if (sum == 1.234567e-30f) C[blockIdx.x * blockDim.x + threadIdx.x] = sum;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("WGMMA m64n64k16 BF16: total_cy=%llu cy/wgmma=%.3f\n",
               t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
