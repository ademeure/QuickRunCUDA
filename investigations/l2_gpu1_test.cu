// Same benchmark but on GPU 1 to see if BW differs
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

#define BLOCK_SIZE 1024
#define BLOCKS_PER_SM 2
#define UNROLL 16

__global__ __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
void bench_cg(const uint32_t* __restrict__ A, uint32_t* __restrict__ C, int ITERS, uint32_t WS) {
    uint32_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t nthr = gridDim.x*blockDim.x;
    uint64_t base = (uint64_t)(uintptr_t)A;
    uint32_t mask = WS-1u;
    uint32_t a=0,b=0,c=0,d=0;
    #pragma unroll 1
    for (int i=0; i<ITERS; i+=UNROLL) {
        #pragma unroll
        for (int j=0; j<UNROLL; j++) {
            uint32_t off = (uint32_t)(((uint64_t)tid*16ULL+(uint64_t)(i+j)*(uint64_t)nthr*16ULL)&mask)&~15u;
            uint64_t addr = base+off;
            uint32_t x0,x1,x2,x3;
            asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];"
                :"=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3):"l"(addr):"memory");
            a^=x0; b^=x1; c^=x2; d^=x3;
        }
    }
    C[tid] = a^b^c^d;
}

#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA err %s\n",cudaGetErrorString(e));exit(1);}}while(0)

int main() {
    for (int gpu = 0; gpu < 2; gpu++) {
        CHECK(cudaSetDevice(gpu));
        int nsm=0;
        CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, gpu));
        
        int nb = nsm * BLOCKS_PER_SM;
        int bs = BLOCK_SIZE;
        long long nthr = (long long)nb * bs;
        
        size_t ws = 32ULL*1024*1024;
        uint32_t *d_A, *d_C;
        CHECK(cudaMalloc(&d_A, ws));
        CHECK(cudaMalloc(&d_C, nthr*4));
        CHECK(cudaMemset(d_A, 0, ws));
        
        int iters = 32768*16;  // 524288 to get long run
        dim3 grid(nb), block(bs);
        
        // Warmup
        for (int w=0;w<3;w++) bench_cg<<<grid,block>>>(d_A,(uint32_t*)d_C,iters,(uint32_t)ws);
        CHECK(cudaDeviceSynchronize());
        
        cudaEvent_t t0,t1;
        CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
        
        float best_ms = 1e9;
        for (int r=0;r<5;r++) {
            CHECK(cudaEventRecord(t0));
            bench_cg<<<grid,block>>>(d_A,(uint32_t*)d_C,iters,(uint32_t)ws);
            CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
            float ms=0; CHECK(cudaEventElapsedTime(&ms,t0,t1));
            if (ms < best_ms) best_ms = ms;
        }
        double bytes = (double)nthr * iters * 16.0;
        double bw = bytes / (best_ms * 1e-3) / 1e12;
        printf("GPU%d: %d SMs, %lld threads, WS=32MB, ITERS=%d -> BW=%.2f TB/s (best of 5, time=%.1f ms)\n",
               gpu, nsm, nthr, iters, bw, best_ms);
        
        CHECK(cudaFree(d_A)); CHECK(cudaFree(d_C));
        CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
    }
    return 0;
}
