// V36: TMA double-buffered (pipelined R+W via 2 buffers)
// Iter N reads into buf[N%2], writes from buf[(N-1)%2]
// Goal: overlap R and W to push closer to A6 6.68 TB/s

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int TILE_BYTES, int N_ITERS>
__global__ __launch_bounds__(128, 1)
void tma_pipelined(const float* src, float* dst, unsigned long long* out, unsigned total_ctas) {
    extern __shared__ __align__(16) char buf_raw[];
    unsigned* buf[2];
    buf[0] = (unsigned*)buf_raw;
    buf[1] = (unsigned*)(buf_raw + TILE_BYTES);
    __shared__ __align__(8) unsigned long long mbar[2];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&mbar[0])) : "memory");
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&mbar[1])) : "memory");
    }
    __syncthreads();

    unsigned buf_addr[2] = {
        (unsigned)__cvta_generic_to_shared(buf[0]),
        (unsigned)__cvta_generic_to_shared(buf[1])
    };
    unsigned mbar_addr[2] = {
        (unsigned)__cvta_generic_to_shared(&mbar[0]),
        (unsigned)__cvta_generic_to_shared(&mbar[1])
    };

    size_t stride = total_ctas * (TILE_BYTES / 4);
    size_t cap = 1ull * 1024 * 1024 * 1024;

    auto issue_read = [&](int slot, size_t off) {
        if (tid == 0) {
            asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbar_addr[slot]) : "memory");
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                         :: "r"(mbar_addr[slot]), "r"(TILE_BYTES) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %2, [%3];\n"
                         :: "r"(buf_addr[slot]), "l"(src + off), "r"(TILE_BYTES), "r"(mbar_addr[slot])
                         : "memory");
        }
    };
    auto wait_read = [&](int slot) {
        if (tid == 0) {
            int done = 0; int spin = 0;
            while (!done && spin < 1000000) {
                asm volatile("{.reg .pred p;\n"
                             "mbarrier.try_wait.shared.b64 p, [%1], 0;\n"
                             "selp.u32 %0, 1, 0, p;}\n"
                             : "=r"(done) : "r"(mbar_addr[slot]) : "memory");
                spin++;
            }
        }
    };
    auto issue_write = [&](int slot, size_t off) {
        if (tid == 0) {
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                         :: "l"(dst + off), "r"(buf_addr[slot]), "r"(TILE_BYTES)
                         : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
    };
    auto wait_write = [&]() {
        if (tid == 0) {
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
    };

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Prefetch first tile into slot 0
    size_t off0 = (bid * (size_t)(TILE_BYTES / 4)) % (cap - TILE_BYTES/4);
    issue_read(0, off0);

    #pragma unroll 1
    for (int it = 1; it < N_ITERS; it++) {
        int prev = (it - 1) & 1;
        int curr = it & 1;

        size_t off = (bid * (size_t)(TILE_BYTES / 4) + (size_t)it * stride) % (cap - TILE_BYTES/4);
        size_t off_prev = (bid * (size_t)(TILE_BYTES / 4) + (size_t)(it-1) * stride) % (cap - TILE_BYTES/4);

        // Wait for prev read complete
        wait_read(prev);
        __syncthreads();

        // Issue write of prev (smem→global) AND read of curr (global→smem) — should overlap
        issue_write(prev, off_prev);
        issue_read(curr, off);
        // Don't wait for write here — let it overlap with next read
    }

    // Drain: wait for last read, write it, drain writes
    int last = (N_ITERS - 1) & 1;
    size_t off_last = (bid * (size_t)(TILE_BYTES / 4) + (size_t)(N_ITERS-1) * stride) % (cap - TILE_BYTES/4);
    wait_read(last);
    __syncthreads();
    issue_write(last, off_last);
    wait_write();
    __syncthreads();

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0 && bid == 0) {
        out[0] = t1 - t0;
        ((float*)&out[2])[0] = ((float*)buf[0])[0];
    }
}

int main() {
    CK(cudaSetDevice(0));

    size_t words = 1ull * 1024 * 1024 * 1024;
    float *d_src, *d_dst;
    CK(cudaMalloc(&d_src, words * 4));
    CK(cudaMalloc(&d_dst, words * 4));
    cudaMemset(d_src, 0, words * 4);
    cudaMemset(d_dst, 0, words * 4);

    unsigned long long* d_out;
    CK(cudaMalloc(&d_out, 256));

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 512;

    printf("=== V36 TMA pipelined R+W (double-buffer) ===\n");
    printf("tile_KB  n_blocks  wall_ms   R+W_bytes_TB/s  per_iter_us\n");

    auto run = [&](int tile, int blocks) {
        int shmem = tile * 2;
        cudaFuncSetAttribute(tma_pipelined<16384, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_pipelined<32768, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
        cudaFuncSetAttribute(tma_pipelined<65536, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);

        if (tile == 16384) tma_pipelined<16384, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
        else if (tile == 32768) tma_pipelined<32768, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
        else tma_pipelined<65536, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) { printf("%d      %4d      FAIL\n", tile/1024, blocks); return; }

        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            if (tile == 16384) tma_pipelined<16384, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
            else if (tile == 32768) tma_pipelined<32768, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
            else tma_pipelined<65536, 512><<<blocks, 128, shmem>>>((const float*)d_src, d_dst, d_out, blocks);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        double rw_bytes = (double)blocks * N_ITERS * tile * 2;
        double rw_tbs = rw_bytes / (avg_ms / 1e3) / 1e12;
        unsigned long long cy;
        cudaMemcpy(&cy, d_out, 8, cudaMemcpyDeviceToHost);
        double per_iter_us = cy / (double)N_ITERS / 1.920 / 1e3;
        printf("%d      %4d      %7.3f   %12.3f      %5.2f\n", tile/1024, blocks, avg_ms, rw_tbs, per_iter_us);
    };

    for (int b : {64, 128, 148}) run(16384, b);
    for (int b : {64, 128, 148}) run(32768, b);
    for (int b : {64, 128, 148}) run(65536, b);

    printf("\nHBM R+W peak (A6 50:50): 6.68 TB/s. V35 sequential: 6.11 TB/s.\n");

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_out);
    return 0;
}
