// Pipelined cp.async with overlap
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

extern "C" __global__ void cpasync_pipe(const int4 *src, int *out, int N, int iters) {
    extern __shared__ int4 smem[];  // 2 stages
    int tid = threadIdx.x;

    // Prime stage 0
    if (tid < 32) {
        uint32_t addr = __cvta_generic_to_shared(&smem[tid]);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :: "r"(addr), "l"(src + tid));
    }
    asm volatile("cp.async.commit_group;");

    int sum = 0;
    for (int it = 0; it < iters; it++) {
        // Issue next stage's copy
        int next_stage = (it + 1) & 1;
        if (tid < 32) {
            uint32_t addr = __cvta_generic_to_shared(&smem[next_stage * 32 + tid]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(addr), "l"(src + ((tid + it + 1) & 1023)));
        }
        asm volatile("cp.async.commit_group;");

        // Wait for THIS stage's copy
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        // Use current stage data
        int cur_stage = it & 1;
        sum += smem[cur_stage * 32 + (tid & 31)].x;
    }
    if (sum < -1) out[blockIdx.x] = sum;
}

extern "C" __global__ void sync_load_store(const int4 *src, int *out, int N, int iters) {
    extern __shared__ int4 smem[];
    int tid = threadIdx.x;
    int sum = 0;
    for (int it = 0; it < iters; it++) {
        if (tid < 32) {
            smem[tid] = src[(tid + it) & 1023];
        }
        __syncthreads();
        sum += smem[tid & 31].x;
    }
    if (sum < -1) out[blockIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);

    int N = 1024 * 16;
    int4 *d_src; cudaMalloc(&d_src, N * sizeof(int4));
    cudaMemset(d_src, 1, N * sizeof(int4));
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 32;

    auto bench = [&](auto launch, size_t smem_bytes, const char *name) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total = (long)blocks * 32 * iters * 16;
        double tb = total / (best/1000.0) / 1e12;
        printf("  %-30s %.3f ms  %.2f TB/s\n", name, best, tb);
    };

    printf("# B300 pipelined cp.async vs sync load+store\n\n");
    bench([&]{ cpasync_pipe<<<blocks, threads, 2*32*sizeof(int4)>>>(d_src, d_out, N, iters); }, 2*32*sizeof(int4), "cp.async pipelined (2 stage)");
    bench([&]{ sync_load_store<<<blocks, threads, 32*sizeof(int4)>>>(d_src, d_out, N, iters); }, 32*sizeof(int4), "sync load+store");

    return 0;
}
