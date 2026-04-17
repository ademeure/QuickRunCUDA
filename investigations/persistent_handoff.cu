// Persistent kernel - careful version
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void persistent_worker(
    int *cmd_idx,
    int *done_idx,
    int *work_buf,
    int n_iter,
    int max_idx
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int next = 0;
        while (next < n_iter) {
            int cur;
            do {
                asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(cur) : "l"(cmd_idx));
            } while (cur <= next);

            int item = work_buf[next % max_idx];
            int result = item * 2 + 1;
            work_buf[next % max_idx] = result;

            asm volatile("st.release.sys.u32 [%0], %1;" :: "l"(done_idx), "r"(next + 1));
            next++;
        }
    }
}

int main() {
    cudaSetDevice(0);

    int *d_work; cudaMalloc(&d_work, 1024 * sizeof(int));
    int *h_cmd, *h_done;
    cudaHostAlloc(&h_cmd, sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc(&h_done, sizeof(int), cudaHostAllocMapped);

    int *d_cmd, *d_done;
    cudaHostGetDevicePointer(&d_cmd, h_cmd, 0);
    cudaHostGetDevicePointer(&d_done, h_done, 0);

    cudaStream_t s; cudaStreamCreate(&s);

    printf("# B300 persistent kernel handoff (with sys-scope load/store)\n\n");

    for (int n_work : {100, 1000, 10000}) {
        *h_cmd = 0;
        *h_done = 0;

        persistent_worker<<<1, 32, 0, s>>>(d_cmd, d_done, d_work, n_work, 256);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_work; i++) {
            volatile int *vh_cmd = (volatile int *)h_cmd;
            volatile int *vh_done = (volatile int *)h_done;
            __sync_synchronize();
            *vh_cmd = i + 1;
            while (*vh_done < i + 1) {}
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        cudaStreamSynchronize(s);
        float us = std::chrono::duration<float, std::micro>(t1-t0).count();
        printf("  %-8d items: %.1f us total = %.2f us per round-trip\n",
               n_work, us, us/n_work);
    }

    return 0;
}
