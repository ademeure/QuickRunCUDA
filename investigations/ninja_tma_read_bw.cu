// TMA bulk read BW via cuda::memcpy_async vs LDG.E.128 (7365 GB/s baseline)
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

namespace cg = cooperative_groups;
using barrier = cuda::barrier<cuda::thread_scope_block>;

template <int CHUNK_BYTES>
__launch_bounds__(128, 4) __global__ void tma_read(const char *__restrict__ gmem, int *sink, int N_iters) {
    extern __shared__ __align__(16) char smem[];
    __shared__ barrier bar;
    auto block = cg::this_thread_block();
    if (threadIdx.x == 0) init(&bar, block.size());
    block.sync();

    size_t base = (size_t)blockIdx.x * (size_t)CHUNK_BYTES * (size_t)N_iters;
    int sum = 0;
    for (int i = 0; i < N_iters; i++) {
        const char *g = gmem + base + (size_t)i * CHUNK_BYTES;
        cuda::memcpy_async(block, smem, g, cuda::aligned_size_t<16>(CHUNK_BYTES), bar);
        bar.arrive_and_wait();
        sum += *((int*)smem + (threadIdx.x & ((CHUNK_BYTES/4) - 1)));
    }
    if (sum == 0xdeadbeef) sink[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 4ull * 1024 * 1024 * 1024;
    char *d_data;
    cudaMalloc((void**)&d_data, bytes);
    cudaMemset(d_data, 0xab, bytes);
    int *d_sink;
    cudaMalloc((void**)&d_sink, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto run = [&](const char* name, int chunk, void(*kfn)(const char*, int*, int), int blocks) {
        int N_iters = bytes / ((size_t)blocks * chunk);
        if (N_iters < 1) { printf("  %s: too few iters\n", name); return; }
        size_t total = (size_t)blocks * N_iters * chunk;
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
        for (int i = 0; i < 3; i++) kfn<<<blocks, 128, chunk>>>(d_data, d_sink, N_iters);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  %s: ERROR %s\n", name, cudaGetErrorString(err));
            return;
        }
        float best = 1e30f;
        for (int i = 0; i < 10; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, 128, chunk>>>(d_data, d_sink, N_iters);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double gbs = total / (best/1000.0) / 1e9;
        printf("  %-30s chunk=%5d blocks=%5d N_iters=%5d  %.4f ms = %.0f GB/s = %.2f%% spec\n",
               name, chunk, blocks, N_iters, best, gbs, gbs/7672*100);
    };

    // chunk × blocks sweep
    run("TMA chunk=512",   512,   tma_read<512>,   37888);
    run("TMA chunk=1024",  1024,  tma_read<1024>,  37888);
    run("TMA chunk=2048",  2048,  tma_read<2048>,  18944);
    run("TMA chunk=4096",  4096,  tma_read<4096>,  9472);
    run("TMA chunk=8192",  8192,  tma_read<8192>,  4736);
    run("TMA chunk=16384", 16384, tma_read<16384>, 2368);
    run("TMA chunk=32768", 32768, tma_read<32768>, 1184);

    printf("\n# Sweep blocks at chunk=8192:\n");
    for (int b : {148, 296, 592, 1184, 2368, 4736, 9472, 18944, 37888})
        run("TMA chunk=8192", 8192, tma_read<8192>, b);
    return 0;
}
