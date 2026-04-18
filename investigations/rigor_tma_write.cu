// TMA bulk write loop with memory clobber
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

__launch_bounds__(128, 6) __global__ void tma_loop(int4 *data, long total_chunks) {
    extern __shared__ int4 smem[];
    int tid = threadIdx.x;

    int4 v = make_int4(0xab, 0xab, 0xab, 0xab);
    for (int i = tid; i < 512; i += blockDim.x) smem[i] = v;
    __syncthreads();

    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    long my_chunks_per_block = total_chunks / gridDim.x;
    long blk = blockIdx.x;

    for (long c = 0; c < my_chunks_per_block; c++) {
        int4 *dst = data + ((blk * my_chunks_per_block + c) * 512);  // 8KB / 16 = 512 int4
        if (tid == 0) {
            asm volatile(
                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 8192;\n"
                :: "l"(dst), "r"(smem_addr) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
    }
    if (tid == 0) {
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
    __syncthreads();
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    size_t bytes = 4096ul * 1024 * 1024;
    int4 *d; cudaMalloc(&d, bytes);
    cudaMemset(d, 0, bytes);  // pre-fill 0

    long total_chunks = bytes / 8192;  // 524288 chunks
    int blocks = 148 * 6;  // 888 blocks
    int threads = 128;

    cudaFuncSetAttribute(tma_loop, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192);

    auto bench = [&](auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaError_t err = cudaDeviceSynchronize();
        if (err) { printf("ERR: %s\n", cudaGetErrorString(err)); return -1.0f; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    float t = bench([&]{ tma_loop<<<blocks, threads, 8192>>>(d, total_chunks); });
    if (t > 0) {
        double bw = bytes/(t/1000)/1e9;
        printf("# TMA bulk loop: 524288 × 8KB = 4 GB, 888 blocks × 128 thr\n");
        printf("  time:  %.3f ms\n", t);
        printf("  BW:    %.0f GB/s (%.1f%% of 7672 peak)\n", bw, bw/7672*100);
    }

    // Verify some bytes
    int h[10];
    cudaMemcpy(h, (int*)d + 1000000, 40, cudaMemcpyDeviceToHost);
    printf("\nVerify (read back at offset 1M ints, all should be 0xab=171):\n  ");
    for (int i = 0; i < 10; i++) printf("%d ", h[i]);
    printf("\n");

    return 0;
}
