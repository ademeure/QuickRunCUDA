#include <cuda_runtime.h>
#include <cstdio>

__global__ void int8_mma(int *out, int iters) {
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }

    if (c0+c1+c2+c3 == 0xdeadbeef) out[blockIdx.x*blockDim.x+threadIdx.x] = c0+c1+c2+c3;
}

__global__ void int8_mma_nosat(int *out, int iters) {
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }

    if (c0+c1+c2+c3 == 0xdeadbeef) out[blockIdx.x*blockDim.x+threadIdx.x] = c0+c1+c2+c3;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 148 * 256 * sizeof(int));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 256;

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
        int warps = blocks * (threads/32);
        long total = (long)warps * iters;
        long ops = total * 16 * 8 * 32 * 2;
        double tops = ops / (best/1000.0) / 1e12;
        printf("  %-25s %.3f ms = %.0f TOPS\n", name, best, tops);
    };

    printf("# B300 INT8 mma.sync m16n8k32\n\n");
    bench([&]{ int8_mma<<<blocks, threads>>>(d_out, iters); }, "satfinite");
    bench([&]{ int8_mma_nosat<<<blocks, threads>>>(d_out, iters); }, "no satfinite");

    return 0;
}
