// cp.async — asynchronous global → shared copy
#include <cstdint>
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void cpasync_test(const int4 *src, int *out, int N, int iters) {
    extern __shared__ int4 smem[];
    int tid = threadIdx.x;

    int sum = 0;
    for (int it = 0; it < iters; it++) {
        // Issue async copy of one int4 (16 bytes) per thread
        if (tid < 32) {
            uint32_t smem_addr = __cvta_generic_to_shared(&smem[tid]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                :: "r"(smem_addr), "l"(src + ((tid + it) & 1023)));
        }
        // Commit and wait
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        // Use the data
        sum += smem[tid & 31].x;
    }
    if (sum < -1) out[blockIdx.x] = sum;
}

extern "C" __global__ void load_then_store(const int4 *src, int *out, int N, int iters) {
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
    size_t smem_bytes = 32 * sizeof(int4);

    auto bench = [&](auto launch, const char *name) {
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
        long total_bytes = (long)blocks * 32 * iters * 16;
        double tb = total_bytes / (best/1000.0) / 1e12;
        printf("  %-25s %.3f ms  %.2f TB/s\n", name, best, tb);
    };

    printf("# B300 cp.async vs sync load (small SHMEM, %d iter)\n\n", iters);
    bench([&]{ cpasync_test<<<blocks, threads, smem_bytes>>>(d_src, d_out, N, iters); }, "cp.async + commit + wait");
    bench([&]{ load_then_store<<<blocks, threads, smem_bytes>>>(d_src, d_out, N, iters); }, "load + store + sync");

    return 0;
}
