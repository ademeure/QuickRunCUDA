// V42 (B4): TMA + prefetch.L2 combo
// Hypothesis: prefetch.L2 issued ahead of TMA pulls data into L2 → TMA hit rate up
//   V6 I3 found 1.58× speedup for cp.async (non-bulk). Does it apply to bulk?

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int TILE_BYTES, int N_ITERS, int USE_PREFETCH>
__global__ __launch_bounds__(128, 1)
void tma_with_prefetch(const float* src, unsigned long long* out, unsigned total_ctas) {
    extern __shared__ __align__(16) char buf[];
    __shared__ __align__(8) unsigned long long mbar;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid == 0) asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                               :: "r"((unsigned)__cvta_generic_to_shared(&mbar)) : "memory");
    __syncthreads();

    unsigned buf_addr = (unsigned)__cvta_generic_to_shared(&buf[0]);
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    size_t stride = total_ctas * (TILE_BYTES / 4);
    size_t cap = 1ull * 1024 * 1024 * 1024;

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        size_t off_curr = (bid * (size_t)(TILE_BYTES / 4) + (size_t)it * stride) % (cap - TILE_BYTES/4);
        size_t off_next = (bid * (size_t)(TILE_BYTES / 4) + (size_t)(it+1) * stride) % (cap - TILE_BYTES/4);
        const float* my_src = src + off_curr;
        const float* next_src = src + off_next;

        if (tid == 0) {
            asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbar_addr) : "memory");
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                         :: "r"(mbar_addr), "r"(TILE_BYTES) : "memory");

            // Prefetch NEXT tile to L2 if requested (1 prefetch per 256 bytes; emit several)
            if (USE_PREFETCH) {
                #pragma unroll
                for (int p = 0; p < TILE_BYTES; p += 256) {
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(next_src + p/4) : "memory");
                }
            }

            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %2, [%3];\n"
                         :: "r"(buf_addr), "l"(my_src), "r"(TILE_BYTES), "r"(mbar_addr)
                         : "memory");
        }

        if (tid == 0) {
            int done = 0; int spin = 0;
            while (!done && spin < 1000000) {
                asm volatile("{.reg .pred p;\n"
                             "mbarrier.try_wait.shared.b64 p, [%1], 0;\n"
                             "selp.u32 %0, 1, 0, p;}\n"
                             : "=r"(done) : "r"(mbar_addr) : "memory");
                spin++;
            }
        }
        __syncthreads();
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0 && bid == 0) {
        out[0] = t1 - t0;
        ((float*)&out[2])[0] = ((float*)buf)[0];
    }
}

int main() {
    CK(cudaSetDevice(0));

    size_t words = 1ull * 1024 * 1024 * 1024;
    float* d_src;
    CK(cudaMalloc(&d_src, words * 4));
    cudaMemset(d_src, 0, words * 4);

    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 256;

    printf("=== V42 TMA + prefetch.L2 combo ===\n");
    printf("Hypothesis: prefetch.L2 of NEXT tile while TMA loads CURRENT — does L2 warm help?\n\n");
    printf("tile_KB  blocks  variant         wall_ms    HBM_TB/s  per_iter_us\n");

    auto run = [&](int tile, int blocks, int prefetch) {
        cudaFuncSetAttribute(tma_with_prefetch<16384, 256, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_with_prefetch<16384, 256, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_with_prefetch<32768, 256, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_with_prefetch<32768, 256, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_with_prefetch<65536, 256, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_with_prefetch<65536, 256, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);

        const char* lbl = prefetch ? "with-prefetch  " : "no-prefetch    ";
        // warmup
        if (tile == 16384 && prefetch == 0) tma_with_prefetch<16384, 256, 0><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
        if (tile == 16384 && prefetch == 1) tma_with_prefetch<16384, 256, 1><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
        if (tile == 32768 && prefetch == 0) tma_with_prefetch<32768, 256, 0><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
        if (tile == 32768 && prefetch == 1) tma_with_prefetch<32768, 256, 1><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
        if (tile == 65536 && prefetch == 0) tma_with_prefetch<65536, 256, 0><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
        if (tile == 65536 && prefetch == 1) tma_with_prefetch<65536, 256, 1><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("%d      %4d   %s FAIL\n", tile/1024, blocks, lbl); return; }

        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (tile == 16384 && prefetch == 0) tma_with_prefetch<16384, 256, 0><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
            if (tile == 16384 && prefetch == 1) tma_with_prefetch<16384, 256, 1><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
            if (tile == 32768 && prefetch == 0) tma_with_prefetch<32768, 256, 0><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
            if (tile == 32768 && prefetch == 1) tma_with_prefetch<32768, 256, 1><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
            if (tile == 65536 && prefetch == 0) tma_with_prefetch<65536, 256, 0><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
            if (tile == 65536 && prefetch == 1) tma_with_prefetch<65536, 256, 1><<<blocks, 128, tile>>>((const float*)d_src, d_out, blocks);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double bytes = (double)blocks * N_ITERS * tile;
        double tbs = bytes / (avg_ms / 1e3) / 1e12;
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        double per_iter_us = cy / (double)N_ITERS / 1.920 / 1e3;
        printf("%d      %4d   %s %7.3f   %8.3f   %5.2f\n", tile/1024, blocks, lbl, avg_ms, tbs, per_iter_us);
    };

    for (int b : {64, 128, 148}) {
        run(16384, b, 0);
        run(16384, b, 1);
    }
    for (int b : {64, 128, 148}) {
        run(32768, b, 0);
        run(32768, b, 1);
    }
    for (int b : {64, 128, 148}) {
        run(65536, b, 0);
        run(65536, b, 1);
    }

    cudaFree(d_src);
    cudaFree(d_out);
    return 0;
}
