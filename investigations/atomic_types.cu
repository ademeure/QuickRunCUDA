// Different atomic operation type costs
#include <cuda_runtime.h>
#include <cstdio>

template<typename Op>
__global__ void atomic_chain(unsigned *target, unsigned long long *out, int iters, Op op) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        for (int i = 0; i < iters; i++) {
            op(target, i + 1);
        }
        unsigned long long t1 = clock64();
        out[0] = t1 - t0;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned *d_target; cudaMalloc(&d_target, sizeof(unsigned));
    unsigned long long *d_out; cudaMalloc(&d_out, sizeof(unsigned long long));

    int iters = 1000;

    auto run = [&](auto op, const char *name) {
        cudaMemset(d_target, 0, sizeof(unsigned));
        atomic_chain<<<1, 32>>>(d_target, d_out, iters, op);
        cudaDeviceSynchronize();
        unsigned long long cyc; cudaMemcpy(&cyc, d_out, sizeof(cyc), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-25s %.1f cyc = %.2f ns\n", name, per, per/2.032);
    };

    printf("# B300 atomic operation type costs (single thread, 1000 iter, hot location)\n\n");

    run([] __device__ (unsigned *p, unsigned v) { atomicAdd(p, v); }, "atomicAdd");
    run([] __device__ (unsigned *p, unsigned v) { atomicSub(p, v); }, "atomicSub");
    run([] __device__ (unsigned *p, unsigned v) { atomicMin(p, v); }, "atomicMin");
    run([] __device__ (unsigned *p, unsigned v) { atomicMax(p, v); }, "atomicMax");
    run([] __device__ (unsigned *p, unsigned v) { atomicExch(p, v); }, "atomicExch");
    run([] __device__ (unsigned *p, unsigned v) { atomicAnd(p, v); }, "atomicAnd");
    run([] __device__ (unsigned *p, unsigned v) { atomicOr(p, v); }, "atomicOr");
    run([] __device__ (unsigned *p, unsigned v) { atomicXor(p, v); }, "atomicXor");
    run([] __device__ (unsigned *p, unsigned v) { atomicInc(p, 0xffffffff); }, "atomicInc");
    run([] __device__ (unsigned *p, unsigned v) { atomicDec(p, 0xffffffff); }, "atomicDec");
    run([] __device__ (unsigned *p, unsigned v) { atomicCAS(p, 0, v); }, "atomicCAS");

    return 0;
}
