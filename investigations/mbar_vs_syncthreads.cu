#include <cuda_runtime.h>
#include <cstdio>

__global__ void mbar_arrive_wait(unsigned long long *out, int iters) {
    __shared__ unsigned long long mbar;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)), "r"(blockDim.x));
    }
    __syncthreads();

    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
        asm volatile(
            "{\n"
            ".reg .b64 state;\n"
            ".reg .pred p;\n"
            "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
            "WAITX_%=: mbarrier.test_wait.shared::cta.b64 p, [%0], state;\n"
            "@!p bra WAITX_%=;\n"
            "}\n"
            :: "r"((unsigned)__cvta_generic_to_shared(&mbar)));
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

__global__ void syncthreads_loop(unsigned long long *out, int iters) {
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; i++) __syncthreads();
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out;
    cudaMalloc(&d_out, 16*sizeof(unsigned long long));
    int iters = 1000;

    printf("# B300 mbarrier vs syncthreads (1000 iter loop)\n");
    printf("# %-25s %-10s %-12s %-12s\n", "primitive", "threads", "cycles", "ns@2032");

    for (int threads : {32, 64, 128, 256, 512, 1024}) {
        syncthreads_loop<<<1, threads>>>(d_out, iters);
        cudaDeviceSynchronize();
        unsigned long long cyc;
        cudaMemcpy(&cyc, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-25s %-10d %-12.1f %-12.1f\n", "__syncthreads", threads, per, per/2.032);
    }

    printf("\n");
    for (int threads : {32, 64, 128, 256, 512, 1024}) {
        mbar_arrive_wait<<<1, threads>>>(d_out, iters);
        cudaDeviceSynchronize();
        unsigned long long cyc;
        cudaMemcpy(&cyc, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-25s %-10d %-12.1f %-12.1f\n", "mbar.arrive+test_wait", threads, per, per/2.032);
    }
    return 0;
}
