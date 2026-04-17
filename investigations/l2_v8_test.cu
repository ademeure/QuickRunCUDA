// Test with v8.u32 (256-bit) loads for maximum MLP
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

#define BLOCK_SIZE 1024
#define BLOCKS_PER_SM 2
#define UNROLL 8

__global__ __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
void bench_v8_cg(const uint32_t* __restrict__ A, uint32_t* __restrict__ C, int ITERS, uint32_t WS) {
    uint32_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t nthr = gridDim.x*blockDim.x;
    uint64_t base = (uint64_t)(uintptr_t)A;
    uint32_t mask = WS-1u;
    uint32_t a=0,b=0,c=0,d=0,e=0,f=0,g=0,h=0;
    #pragma unroll 1
    for (int i=0; i<ITERS; i+=UNROLL) {
        #pragma unroll
        for (int j=0; j<UNROLL; j++) {
            // 32B per load
            uint32_t off = (uint32_t)(((uint64_t)tid*32ULL+(uint64_t)(i+j)*(uint64_t)nthr*32ULL)&mask)&~31u;
            uint64_t addr = base+off;
            uint32_t x0,x1,x2,x3,x4,x5,x6,x7;
            asm volatile("ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                :"=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3),"=r"(x4),"=r"(x5),"=r"(x6),"=r"(x7)
                :"l"(addr):"memory");
            a^=x0; b^=x1; c^=x2; d^=x3; e^=x4; f^=x5; g^=x6; h^=x7;
        }
    }
    C[tid] = a^b^c^d^e^f^g^h;
}

#define CHECK(x) do{cudaError_t err=(x);if(err!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(err));exit(1);}}while(0)

int main() {
    CHECK(cudaSetDevice(0));
    int nsm=0;
    CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    int nb=nsm*BLOCKS_PER_SM, bs=BLOCK_SIZE;
    long long nthr=(long long)nb*bs;
    
    size_t ws=32ULL*1024*1024;
    uint32_t *d_A,*d_C;
    CHECK(cudaMalloc(&d_A, ws));
    CHECK(cudaMalloc(&d_C, nthr*4));
    CHECK(cudaMemset(d_A, 0, ws));
    
    int iters = 32768 * 8;  // same total bytes as v4 with ITERS=32768*16
    dim3 grid(nb), block(bs);
    
    for (int w=0;w<3;w++) bench_v8_cg<<<grid,block>>>(d_A,(uint32_t*)d_C,iters,(uint32_t)ws);
    CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t t0,t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    
    double best=0;
    for (int r=0;r<5;r++) {
        CHECK(cudaEventRecord(t0));
        bench_v8_cg<<<grid,block>>>(d_A,(uint32_t*)d_C,iters,(uint32_t)ws);
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
        float ms=0; CHECK(cudaEventElapsedTime(&ms,t0,t1));
        double bw=(double)nthr*iters*32.0/(ms*1e-3)/1e12;
        if (bw>best) best=bw;
    }
    printf("v8.cg, 32MB WS, %d blocks x %d threads = %lld total, ITERS=%d -> BW=%.2f TB/s\n",
           nb, bs, nthr, iters, best);
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_C));
    return 0;
}
