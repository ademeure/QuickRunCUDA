// Minimal probe for ncu - runs one configuration at a time
// Launch: ./l2_ncu_probe <WS_MB> <use_cg>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define BLOCK_SIZE 1024
#define BLOCKS_PER_SM 2
#define UNROLL 16

__global__ __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
void bench_cg(const uint32_t* __restrict__ A, uint32_t* __restrict__ C,
              int ITERS, uint32_t WS_BYTES) {
    uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nthr = gridDim.x * blockDim.x;
    uint32_t mask = WS_BYTES - 1u;
    uint64_t base = (uint64_t)(uintptr_t)A;
    uint32_t acc0=0, acc1=0, acc2=0, acc3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            uint32_t off = (uint32_t)(((uint64_t)tid*16ULL + (uint64_t)(i+j)*(uint64_t)nthr*16ULL) & mask) & ~15u;
            uint64_t addr = base + off;
            uint32_t x0,x1,x2,x3;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
                :"=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3):"l"(addr):"memory");
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    C[tid] = acc0^acc1^acc2^acc3;
}

__global__ __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
void bench_ca(const uint32_t* __restrict__ A, uint32_t* __restrict__ C,
              int ITERS, uint32_t WS_BYTES) {
    uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nthr = gridDim.x * blockDim.x;
    uint32_t mask = WS_BYTES - 1u;
    uint64_t base = (uint64_t)(uintptr_t)A;
    uint32_t acc0=0, acc1=0, acc2=0, acc3=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            uint32_t off = (uint32_t)(((uint64_t)tid*16ULL + (uint64_t)(i+j)*(uint64_t)nthr*16ULL) & mask) & ~15u;
            uint64_t addr = base + off;
            uint32_t x0,x1,x2,x3;
            asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3},[%4];"
                :"=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3):"l"(addr):"memory");
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    C[tid] = acc0^acc1^acc2^acc3;
}

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA err %s\n",cudaGetErrorString(e));exit(1);}} while(0)

int main(int argc, char** argv) {
    int ws_mb = (argc > 1) ? atoi(argv[1]) : 32;
    int use_cg = (argc > 2) ? atoi(argv[2]) : 1;

    CHECK(cudaSetDevice(0));
    int nsm=0;
    CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));

    int nb = nsm * BLOCKS_PER_SM;
    int bs = BLOCK_SIZE;
    long long nthr = (long long)nb * bs;

    size_t ws = (size_t)ws_mb * 1024 * 1024;
    size_t c_sz = (size_t)nthr * 4;

    uint32_t *d_A, *d_C;
    CHECK(cudaMalloc(&d_A, ws));
    CHECK(cudaMalloc(&d_C, c_sz));
    CHECK(cudaMemset(d_A, 0, ws));
    CHECK(cudaMemset(d_C, 0, c_sz));

    // Choose ITERS for ~100ms run
    double target = 200e12;  // 200 TB
    int iters = (int)(target / ((double)nthr * 16.0));
    iters = ((iters + UNROLL-1)/UNROLL)*UNROLL;
    if (iters < UNROLL) iters = UNROLL;

    uint32_t ws32 = (uint32_t)ws;
    dim3 grid(nb), block(bs);

    // Warmup
    for (int w = 0; w < 3; w++) {
        if (use_cg) bench_cg<<<grid,block>>>(d_A, d_C, iters, ws32);
        else        bench_ca<<<grid,block>>>(d_A, d_C, iters, ws32);
    }
    CHECK(cudaDeviceSynchronize());

    // Timed
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    CHECK(cudaEventRecord(t0));
    if (use_cg) bench_cg<<<grid,block>>>(d_A, d_C, iters, ws32);
    else        bench_ca<<<grid,block>>>(d_A, d_C, iters, ws32);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms=0;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));

    double bytes = (double)nthr * (double)iters * 16.0;
    double bw = bytes / (ms * 1e-3) / 1e12;

    printf("WS=%d MB  hint=%s  blocks=%d bs=%d iters=%d  BW=%.2f TB/s  time=%.1f ms\n",
           ws_mb, use_cg?"cg":"ca", nb, bs, iters, bw, ms);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    return 0;
}
