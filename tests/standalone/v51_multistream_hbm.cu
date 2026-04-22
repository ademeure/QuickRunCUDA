// V51: Multi-stream HBM — can N streams aggregate exceed single-stream peak?
// Theoretical: HBM is shared physical resource → aggregate ≤ 7.31 TB/s

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int TILE_BYTES, int N_INFLIGHT, int N_ITERS>
__global__ __launch_bounds__(128, 1)
void tma_8deep(const float* src, unsigned long long* out, unsigned total_ctas) {
    extern __shared__ __align__(16) char buf_raw[];
    __shared__ __align__(8) unsigned long long mbar[16];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < N_INFLIGHT; i++) {
            asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                         :: "r"((unsigned)__cvta_generic_to_shared(&mbar[i])) : "memory");
        }
    }
    __syncthreads();

    unsigned bufs[16];
    unsigned mbars[16];
    #pragma unroll
    for (int i = 0; i < N_INFLIGHT; i++) {
        bufs[i] = (unsigned)__cvta_generic_to_shared(&buf_raw[i * TILE_BYTES]);
        mbars[i] = (unsigned)__cvta_generic_to_shared(&mbar[i]);
    }

    size_t stride = total_ctas * (TILE_BYTES / 4);
    size_t cap = 1ull * 1024 * 1024 * 1024;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        if (tid == 0) {
            #pragma unroll
            for (int i = 0; i < N_INFLIGHT; i++) {
                asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(mbars[i]) : "memory");
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                             :: "r"(mbars[i]), "r"(TILE_BYTES) : "memory");
                size_t off = (bid * (size_t)(TILE_BYTES / 4) + (size_t)(it * N_INFLIGHT + i) * stride) % (cap - TILE_BYTES/4);
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1], %2, [%3];\n"
                             :: "r"(bufs[i]), "l"(src + off), "r"(TILE_BYTES), "r"(mbars[i])
                             : "memory");
            }
            #pragma unroll
            for (int i = 0; i < N_INFLIGHT; i++) {
                int done = 0; int spin = 0;
                while (!done && spin < 1000000) {
                    asm volatile("{.reg .pred p;\n"
                                 "mbarrier.try_wait.shared.b64 p, [%1], 0;\n"
                                 "selp.u32 %0, 1, 0, p;}\n"
                                 : "=r"(done) : "r"(mbars[i]) : "memory");
                    spin++;
                }
            }
        }
        __syncthreads();
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (tid == 0 && bid == 0) {
        out[0] = t1 - t0;
        ((float*)&out[2])[0] = ((float*)buf_raw)[0];
    }
}

int main() {
    CK(cudaSetDevice(0));

    size_t words = 1ull * 1024 * 1024 * 1024;  // 4 GB per stream
    float* d_src[8];
    for (int s = 0; s < 8; s++) {
        CK(cudaMalloc(&d_src[s], words * 4));
        cudaMemset(d_src[s], 0, words * 4);
    }

    unsigned long long* d_out[8];
    for (int s = 0; s < 8; s++) CK(cudaMalloc(&d_out[s], 256));

    cudaStream_t streams[8];
    for (int s = 0; s < 8; s++) cudaStreamCreate(&streams[s]);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    const int N_ITERS = 256;
    const int TILE = 16384;
    const int N_INFLIGHT = 8;
    const int BLOCKS = 148;
    const int SHMEM = N_INFLIGHT * TILE;

    cudaFuncSetAttribute(tma_8deep<TILE, N_INFLIGHT, N_ITERS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);

    printf("=== V51 Multi-stream HBM aggregate ===\n");
    printf("Each stream: 148 blocks × 16 KB × 8-deep × 256 iters (V46 setup)\n");
    printf("Theoretical: HBM peak ≤ 7.31 TB/s regardless of stream count\n\n");
    printf("N_streams  blocks_per_stream  wall_ms   total_TB/s   per_stream_TB/s\n");

    for (int n_streams : {1, 2, 4, 8}) {
        // warmup
        for (int s = 0; s < n_streams; s++) {
            tma_8deep<TILE, N_INFLIGHT, N_ITERS><<<BLOCKS / n_streams, 128, SHMEM, streams[s]>>>(
                (const float*)d_src, d_out[s], BLOCKS / n_streams);
        }
        cudaDeviceSynchronize();

        int RUNS = 5;
        float total_ms = 0;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(e0);
            for (int s = 0; s < n_streams; s++) {
                tma_8deep<TILE, N_INFLIGHT, N_ITERS><<<BLOCKS / n_streams, 128, SHMEM, streams[s]>>>(
                    (const float*)d_src, d_out[s], BLOCKS / n_streams);
            }
            for (int s = 0; s < n_streams; s++) cudaStreamSynchronize(streams[s]);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            total_ms += ms;
        }
        float avg_ms = total_ms / RUNS;
        // Each stream moves: (BLOCKS/n_streams) × N_ITERS × N_INFLIGHT × TILE bytes
        double bytes_per_stream = (double)(BLOCKS / n_streams) * N_ITERS * N_INFLIGHT * TILE;
        double total_bytes = bytes_per_stream * n_streams;
        double total_tbs = total_bytes / (avg_ms / 1e3) / 1e12;
        printf("%d         %5d              %7.3f   %8.3f     %.2f\n",
               n_streams, BLOCKS / n_streams, avg_ms, total_tbs, total_tbs / n_streams);
    }

    for (int s = 0; s < 8; s++) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_out[s]);
    }
    cudaFree(d_src);
    return 0;
}
