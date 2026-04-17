// Measure mbarrier primitive costs on B300
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void mbar_test(unsigned long long *out) {
    __shared__ alignas(8) unsigned long long mbar;
    int tid = threadIdx.x;

    unsigned long long start, end;

    // Test 1: init cost
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
        for (int i = 0; i < 100; i++) {
            asm volatile("mbarrier.init.shared.b64 [%0], 32;" :: "l"(&mbar));
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
        out[0] = end - start;  // 100 init ops
    }
    __syncthreads();

    // Reset mbar
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "l"(&mbar), "r"(32u));
    }
    __syncthreads();

    // Test 2: arrive cost (all 32 threads)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    unsigned long long token;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
                 : "=l"(token) : "l"(&mbar));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[1] = end - start;

    __syncthreads();

    // Reset for wait test
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "l"(&mbar), "r"(32u));
    }
    __syncthreads();

    // Arrive
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
                 : "=l"(token) : "l"(&mbar));
    __syncthreads();

    // Test 3: test_wait cost (should complete since all 32 arrived)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    int done;
    asm volatile("{ .reg .pred p; mbarrier.test_wait.shared.b64 p, [%1], %2; selp.s32 %0, 1, 0, p; }"
                 : "=r"(done) : "l"(&mbar), "l"(token));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) {
        out[2] = end - start;
        out[3] = done;  // should be 1 (all arrived)
    }

    // Reset for try_wait test (with timeout)
    __syncthreads();
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "l"(&mbar), "r"(32u));
    }
    __syncthreads();
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];"
                 : "=l"(token) : "l"(&mbar));
    __syncthreads();

    // Test 4: try_wait (sleep + retry)
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    asm volatile("{ .reg .pred p; mbarrier.try_wait.shared.b64 p, [%1], %2; selp.s32 %0, 1, 0, p; }"
                 : "=r"(done) : "l"(&mbar), "l"(token));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (tid == 0) out[4] = end - start;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out;
    cudaMalloc(&d_out, 16 * sizeof(unsigned long long));

    mbar_test<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();

    unsigned long long h[5];
    cudaMemcpy(h, d_out, 5 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("# B300 mbarrier primitive costs (cycles)\n");
    printf("# (measured via clock64 on single warp)\n\n");
    printf("  mbarrier.init (100×):          %llu cy (%.2f cy each)\n", h[0], h[0] / 100.0);
    printf("  mbarrier.arrive (32 threads):  %llu cy\n", h[1]);
    printf("  mbarrier.test_wait (post-arrive, should pass): %llu cy, done=%llu\n", h[2], h[3]);
    printf("  mbarrier.try_wait (post-arrive):               %llu cy\n", h[4]);

    cudaFree(d_out);
    return 0;
}
