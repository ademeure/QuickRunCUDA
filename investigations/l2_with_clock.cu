// Measures L2 BW and reports the SM clock via globaltimer  
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

#define BLOCK_SIZE 1024
#define BLOCKS_PER_SM 2
#define UNROLL 16

__global__ __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
void bench_cg_clock(const uint32_t* __restrict__ A, uint32_t* __restrict__ C, 
                    int ITERS, uint32_t WS, uint64_t* clk_out) {
    uint32_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t nthr = gridDim.x*blockDim.x;
    uint64_t base = (uint64_t)(uintptr_t)A;
    uint32_t mask = WS-1u;
    uint32_t a=0,b=0,c=0,d=0;
    
    uint64_t clk_start = 0, clk_end = 0;
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(clk_start));
    }
    
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
    
    if (tid == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(clk_end));
        clk_out[blockIdx.x] = clk_end - clk_start;
    }
    C[tid] = a^b^c^d;
}

#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA err %s\n",cudaGetErrorString(e));exit(1);}}while(0)

int main() {
    CHECK(cudaSetDevice(0));
    int nsm=0;
    CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    int nb=nsm*BLOCKS_PER_SM, bs=BLOCK_SIZE;
    long long nthr=(long long)nb*bs;
    
    size_t ws=32ULL*1024*1024;
    uint32_t *d_A,*d_C;
    uint64_t *d_clk;
    CHECK(cudaMalloc(&d_A, ws));
    CHECK(cudaMalloc(&d_C, nthr*4));
    CHECK(cudaMalloc(&d_clk, nb*8));
    CHECK(cudaMemset(d_A, 0, ws));
    
    int iters = 524288;
    dim3 grid(nb), block(bs);
    
    // Warmup
    for (int w=0;w<3;w++) bench_cg_clock<<<grid,block>>>(d_A,d_C,iters,(uint32_t)ws,d_clk);
    CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t t0,t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    
    float best_ms=1e9;
    uint64_t *h_clk = new uint64_t[nb];
    
    for (int r=0;r<5;r++) {
        CHECK(cudaEventRecord(t0));
        bench_cg_clock<<<grid,block>>>(d_A,d_C,iters,(uint32_t)ws,d_clk);
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
        float ms=0; CHECK(cudaEventElapsedTime(&ms,t0,t1));
        if (ms < best_ms) {
            best_ms = ms;
            CHECK(cudaMemcpy(h_clk, d_clk, nb*8, cudaMemcpyDeviceToHost));
        }
    }
    
    // Compute average SM clock from clock64
    uint64_t avg_clk=0;
    for (int i=0;i<nb;i++) avg_clk += h_clk[i];
    avg_clk /= nb;
    
    double bw=(double)nthr*iters*16.0/(best_ms*1e-3)/1e12;
    double wall_time_s = best_ms * 1e-3;
    double sm_freq_mhz = (double)avg_clk / wall_time_s / 1e6;
    
    printf("WS=32MB .cg, %d blocks x %d threads\n", nb, bs);
    printf("Event time: %.2f ms\n", best_ms);
    printf("BW: %.2f TB/s\n", bw);
    printf("Avg SM cycles per run: %lu\n", avg_clk);
    printf("Inferred SM freq from cycles/wall_time: %.1f MHz\n", sm_freq_mhz);
    printf("\nNote: clock64 counts every SM cycle, so freq = cycles/wall_time\n");
    
    delete[] h_clk;
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_C)); CHECK(cudaFree(d_clk));
    return 0;
}
